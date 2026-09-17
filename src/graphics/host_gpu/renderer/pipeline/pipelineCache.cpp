#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", KYTY_GIT_REVISION,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, uint32_t* value) {
	return value != nullptr &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, value, sizeof(*value));
}

// SRT walks evaluate scalar loads eagerly, including ones the shader only reaches under a
// condition (a null acceleration-structure pointer, for instance). Unmapped guest memory
// reads as zero instead of faulting on the host.
bool ReadShaderSrtMemory(void*, uint64_t address, uint32_t* value) {
	if (value == nullptr) {
		return false;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, value, sizeof(*value))) {
		*value = 0;
	}
	return true;
}

// Memory the SRT walk touched while materializing a snapshot. The next draw with the same
// user data replays these reads; if nothing changed the previous snapshot is reused instead of
// walking the descriptor chains again.
struct SrtRead {
	uint64_t address   = 0;
	int64_t  offset    = 0;          // from the base when relocatable
	uint32_t value     = 0;
	uint32_t pair      = UINT32_MAX; // user-data pair the address is based on, or none
	uint32_t base_read = UINT32_MAX; // recorded reads (k, k+1) the address is based on, or none
	bool     clean     = false;      // read through the GPU-clean reader
	bool     ok        = false;
	bool     data      = true;       // the value reached a result and must match on replay
};

struct SrtRecorder {
	std::vector<SrtRead> reads;
	bool                 overflow    = false;
	uint32_t             next_pair   = UINT32_MAX;
	uint32_t             next_base   = UINT32_MAX;
	int64_t              next_offset = 0;
	static constexpr size_t Limit    = ShaderRecompiler::IR::SrtReadSet::MaxReads;

	void Add(uint64_t address, uint32_t value, bool clean, bool ok) {
		if (reads.size() >= Limit) {
			overflow = true;
			return;
		}
		reads.push_back({address, next_offset, value, next_pair, next_base, clean, ok, true});
		next_pair   = UINT32_MAX;
		next_base   = UINT32_MAX;
		next_offset = 0;
	}

	// Marks the reads whose values reached a result; the others only served as bases.
	void FlagData(const ShaderRecompiler::IR::SrtReadSet& data_reads) {
		for (size_t i = 0; i < reads.size(); i++) {
			reads[i].data = data_reads.Contains(static_cast<uint32_t>(i)) ||
			                (reads[i].pair == UINT32_MAX && reads[i].base_read == UINT32_MAX);
		}
	}
};

uint32_t NoteShaderSrtRead(void* context, uint32_t pair, uint32_t base_read, int64_t offset) {
	auto* recorder        = static_cast<SrtRecorder*>(context);
	recorder->next_pair   = pair;
	recorder->next_base   = base_read;
	recorder->next_offset = offset;
	return static_cast<uint32_t>(recorder->reads.size());
}

bool RecordShaderSrtMemory(void* context, uint64_t address, uint32_t* value) {
	const bool ok = ReadShaderSrtMemory(nullptr, address, value);
	static_cast<SrtRecorder*>(context)->Add(address, ok ? *value : 0, false, ok);
	return ok;
}

bool RecordShaderGuestMemory(void* context, uint64_t address, uint32_t* value) {
	const bool ok = ReadShaderGuestMemory(nullptr, address, value);
	static_cast<SrtRecorder*>(context)->Add(address, ok ? *value : 0, true, ok);
	return ok;
}

// KYTY_DEBUG_SRT_MEMO=1: per-shader memo hit/miss counts and walk time, printed every 65536
// lookups (top shaders by walk time, plus the current shader's user data with its mask).
struct SrtMemoStats {
	struct Stat {
		uint64_t hits = 0, misses = 0, walk_ns = 0, prefills = 0, prefill_fails = 0, reads = 0,
		         relocatable = 0, overflow = 0, flagged = 0, sources = 0;
	};
	static std::unordered_map<uint64_t, Stat>& Map() {
		static std::unordered_map<uint64_t, Stat> stats;
		return stats;
	}
	static void AddWalkTime(uint64_t hash, uint64_t ns) { Map()[hash].walk_ns += ns; }
	static void Record(uint64_t hash, bool hit, uint64_t, uint64_t mask,
	                   std::span<const uint32_t> user_data, size_t memos) {
		static uint64_t lookups = 0;
		auto&           stat    = Map()[hash];
		(hit ? stat.hits : stat.misses)++;
		if ((++lookups % 65536) != 0) {
			return;
		}
		std::vector<std::pair<uint64_t, Stat>> sorted(Map().begin(), Map().end());
		std::ranges::sort(sorted, [](const auto& a, const auto& b) {
			return a.second.walk_ns > b.second.walk_ns;
		});
		for (size_t i = 0; i < std::min<size_t>(sorted.size(), 12); i++) {
			const auto& st = sorted[i].second;
			LOGF("SrtMemo: shader=0x%016" PRIx64 " hits=%" PRIu64 " misses=%" PRIu64
			     " walk_ms=%" PRIu64 " prefills=%" PRIu64 " prefill_fails=%" PRIu64 " reads=%" PRIu64
			     " relocatable=%" PRIu64 " overflow=%" PRIu64 " flagged=%" PRIu64 "/%" PRIu64 "\n",
			     sorted[i].first, st.hits, st.misses, st.walk_ns / 1000000, st.prefills,
			     st.prefill_fails, st.reads, st.relocatable, st.overflow, st.flagged, st.sources);
		}
		LOGF("SrtMemo: this shader=0x%016" PRIx64 " mask=0x%" PRIx64 " memos=%zu user_data=%zu\n",
		     hash, mask, memos, user_data.size());
		std::string words;
		for (size_t i = 0; i < user_data.size(); i++) {
			const bool in_mask = i >= 64u || (mask & (uint64_t {1} << i)) != 0;
			words += fmt::format(" {}{:08x}", in_mask ? "*" : "", user_data[i]);
		}
		LOGF("SrtMemo: user data:%s\n", words.c_str());
	}
};

// Replays the recorded reads against the current user data: a relocatable read is re-read at
// the current pointer pair plus its offset, so a snapshot survives the per-draw SRT ring the
// game allocates its descriptor structs from.
// Replays recorded reads against the current user data (relocating them through their bases)
// and reports which ones no longer hold the recorded value. `current`, when given, receives
// the replayed values.
void SrtReplayReads(const std::vector<SrtRead>& reads, std::span<const uint32_t> user_data,
                    ShaderRecompiler::IR::SrtReadSet& mismatches,
                    std::vector<uint32_t>*            current = nullptr) {
	constexpr uint64_t AddressMask = 0x0000ffffffffffffull;
	// Replayed values, for reads that serve as the base of later ones.
	std::array<uint32_t, ShaderRecompiler::IR::SrtReadSet::MaxReads * 2> values {};
	if (current != nullptr) {
		current->assign(reads.size(), 0);
	}
	for (size_t i = 0; i < reads.size(); i++) {
		const auto& read    = reads[i];
		auto        address = read.address;
		bool        based   = true;
		if (read.pair != UINT32_MAX) {
			based = read.pair + 1u < user_data.size();
			if (based) {
				const auto pair_value =
				    (static_cast<uint64_t>(user_data[read.pair]) |
				     (static_cast<uint64_t>(user_data[read.pair + 1]) << 32u)) &
				    AddressMask;
				address = pair_value + static_cast<uint64_t>(read.offset);
			}
		} else if (read.base_read != UINT32_MAX) {
			based = read.base_read + 1u < i && read.base_read + 1u < values.size();
			if (based) {
				const auto base_value =
				    (static_cast<uint64_t>(values[read.base_read]) |
				     (static_cast<uint64_t>(values[read.base_read + 1]) << 32u)) &
				    AddressMask;
				address = base_value + static_cast<uint64_t>(read.offset);
			}
		}
		uint32_t   value = 0;
		const bool ok    = based && (read.clean ? ReadShaderGuestMemory(nullptr, address, &value)
		                                        : ReadShaderSrtMemory(nullptr, address, &value));
		if (!based || ok != read.ok || (read.data && ok && value != read.value)) {
			mismatches.Add(static_cast<uint32_t>(i));
		}
		if (i < values.size()) {
			values[i] = ok ? value : 0;
		}
		if (current != nullptr) {
			(*current)[i] = ok ? value : 0;
		}
	}
}

bool SrtReadsUnchanged(const std::vector<SrtRead>& reads, std::span<const uint32_t> user_data) {
	ShaderRecompiler::IR::SrtReadSet mismatches;
	SrtReplayReads(reads, user_data, mismatches);
	return mismatches.Empty();
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SnapshotMemo {
		bool                                         valid = false;
		uint64_t                                     shader_base = 0;
		std::vector<uint32_t>                        user_data;
		std::vector<SrtRead>                         reads;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
	};

	// Results that depend on guest memory only (through relocatable reads), never on user-data
	// values: the descriptor tables a per-draw SRT struct points at. Validated by replaying the
	// walk's reads at the current pointers and copied into the walk instead of re-evaluated,
	// while the per-draw descriptors embedded in user data are still evaluated.
	struct MemoryMemo {
		// Flags: 0 = evaluated every draw, 1 = prefilled when its reads still match,
		// 2 = retired (its reads kept changing, it is per-draw data).
		bool                                              valid = false;
		uint32_t                                          uses  = 0;
		std::vector<SrtRead>                              reads;
		std::vector<uint8_t>                              source_flags;
		std::vector<uint8_t>                              source_mismatches;
		std::vector<ShaderRecompiler::IR::DescriptorValue> source_values;
		std::vector<ShaderRecompiler::IR::SrtReadSet>      source_reads;
		std::vector<uint8_t>                              flat_flags;
		std::vector<uint8_t>                              flat_mismatches;
		std::vector<uint32_t>                             flat_values;
		std::vector<ShaderRecompiler::IR::SrtReadSet>      flat_reads;
		uint8_t                                           active_flag = 0;
		uint8_t                                           active_mismatches = 0;
		std::vector<uint8_t>                              active_values;
		ShaderRecompiler::IR::SrtReadSet                   active_reads;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan resource_plan;
		std::vector<Permutation>           permutations;
		// Snapshots by user-data content: the same objects are drawn every frame with the same
		// SRT pointers, so most draws replay a previous walk. Only the user-data dwords a walk
		// consumed take part in the key (per-draw constants in the other registers do not change
		// the walk); the mask is the union over all walks and clears the memos when it grows.
		static constexpr size_t                   MemoLimit = 512;
		std::unordered_map<uint64_t, SnapshotMemo> memos;
		uint64_t                                   user_data_mask = 0;
		MemoryMemo                                 memory_memo;
	};

	static bool UserDataInMask(uint64_t mask, size_t index) {
		return index >= 64u || (mask & (uint64_t {1} << index)) != 0;
	}

	static uint64_t HashUserData(uint64_t shader_base, std::span<const uint32_t> user_data,
	                             uint64_t mask) {
		uint64_t hash = shader_base * 0x9E3779B97F4A7C15ull;
		for (size_t index = 0; index < user_data.size(); index++) {
			if (!UserDataInMask(mask, index)) {
				continue;
			}
			hash = (hash ^ user_data[index]) * 0x9E3779B97F4A7C15ull;
			hash ^= hash >> 31u;
		}
		return hash;
	}

	static bool SameMaskedUserData(std::span<const uint32_t> a, std::span<const uint32_t> b,
	                               uint64_t mask) {
		if (a.size() != b.size()) {
			return false;
		}
		for (size_t index = 0; index < a.size(); index++) {
			if (UserDataInMask(mask, index) && a[index] != b[index]) {
				return false;
			}
		}
		return true;
	}

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	Permutation CompilePermutation(const ShaderParams&                          params,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		const char* stage_name = nullptr;
		switch (options.stage) {
			case ShaderType::Vertex: stage_name = "vs"; break;
			case ShaderType::Mesh: stage_name = "ms"; break;
			case ShaderType::Local: stage_name = "ls"; break;
			case ShaderType::TessellationControl: stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: stage_name = "ds"; break;
			case ShaderType::Pixel: stage_name = "ps"; break;
			case ShaderType::Compute: stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code, result.decoded_dump);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		Profiler::ScopedBlock lookup_block(KYTY_PROFILER_SOURCE("Programs::Lookup"));
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = static_cast<uint32_t>(params.user_data.size());
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		lookup_block.End();
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		    .user_data                  = params.user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderSrtMemory,
		    .read_specialization_memory = ReadShaderGuestMemory,
		};
		if (entry != programs.end()) {
			auto&      memos    = entry->second.memos;
			auto       mask     = entry->second.user_data_mask;
			const auto memo_key = HashUserData(params.Base(), params.user_data, mask);
			if (memos.size() >= SourceEntry::MemoLimit) {
				memos.clear();
			}
			static const bool memo_disabled = std::getenv("KYTY_NO_SRT_MEMO") != nullptr;
			bool              memo_hit      = false;
			auto              memo          = memos.find(memo_key);
			{
				KYTY_PROFILER_BLOCK("Programs::MemoCheck");
				memo_hit = !memo_disabled && memo != memos.end() && memo->second.valid &&
				           memo->second.shader_base == params.Base() &&
				           SameMaskedUserData(memo->second.user_data, params.user_data, mask) &&
				           SrtReadsUnchanged(memo->second.reads, params.user_data);
			}
			// KYTY_DEBUG_SRT_MEMO=1: per-shader memo statistics every 65536 lookups.
			static const bool memo_stats = std::getenv("KYTY_DEBUG_SRT_MEMO") != nullptr;
			if (memo_stats) {
				SrtMemoStats::Record(params.hash, memo_hit, 0, mask, params.user_data, memos.size());
			}
			if (memo_hit) {
				KYTY_PROFILER_BLOCK("Programs::MemoHit");
				resources      = memo->second.resources;
				specialization = memo->second.specialization;
				resources.user_data.assign(params.user_data.begin(), params.user_data.end());
			} else {
				KYTY_PROFILER_BLOCK("Programs::Materialize");
				const auto  walk_start = std::chrono::steady_clock::now();
				SrtRecorder recorder;
				uint64_t    walk_mask                                = 0;
				auto        recording_runtime                        = runtime;
				recording_runtime.userdata                           = &recorder;
				recording_runtime.read_memory                        = RecordShaderSrtMemory;
				recording_runtime.read_specialization_memory         = RecordShaderGuestMemory;
				recording_runtime.user_data_mask                     = &walk_mask;
				recording_runtime.note_read                          = NoteShaderSrtRead;
				ShaderRecompiler::IR::SrtReadSet data_reads;
				recording_runtime.data_reads = &data_reads;
				std::vector<uint64_t>                             source_masks;
				std::vector<uint64_t>                             flat_masks;
				std::vector<ShaderRecompiler::IR::SrtReadSet>      source_reads;
				std::vector<ShaderRecompiler::IR::SrtReadSet>      flat_reads;
				std::vector<ShaderRecompiler::IR::DescriptorValue> values;
				std::vector<uint32_t>                             flat;
				recording_runtime.source_masks     = &source_masks;
				recording_runtime.flat_masks       = &flat_masks;
				recording_runtime.source_read_sets = &source_reads;
				recording_runtime.flat_read_sets   = &flat_reads;
				recording_runtime.evaluated_values = &values;
				recording_runtime.evaluated_flat   = &flat;
				ShaderRecompiler::IR::SrtDeps active_deps;
				std::vector<uint8_t>          active_values;
				recording_runtime.active_deps      = &active_deps;
				recording_runtime.evaluated_active = &active_values;
				auto&      memory_memo = entry->second.memory_memo;
				static const bool prefill_disabled = std::getenv("KYTY_NO_SRT_PREFILL") != nullptr;
				std::vector<uint8_t>  prefill_sources;
				std::vector<uint8_t>  prefill_flat;
				std::vector<uint32_t> replayed;
				bool                  prefill = false;
				if (!prefill_disabled && memory_memo.valid && (++memory_memo.uses % 4096) != 0) {
					KYTY_PROFILER_BLOCK("Programs::Prefill");
					ShaderRecompiler::IR::SrtReadSet mismatches;
					SrtReplayReads(memory_memo.reads, params.user_data, mismatches, &replayed);
					prefill_sources.assign(memory_memo.source_flags.size(), 0u);
					prefill_flat.assign(memory_memo.flat_flags.size(), 0u);
					uint32_t candidates = 0;
					uint32_t matched    = 0;
					const auto select = [&](std::vector<uint8_t>& flags, std::vector<uint8_t>& misses,
					                        const std::vector<ShaderRecompiler::IR::SrtReadSet>& sets,
					                        std::vector<uint8_t>& out) {
						for (size_t i = 0; i < flags.size() && i < sets.size(); i++) {
							if (flags[i] != 1u) {
								continue;
							}
							candidates++;
							if (!sets[i].Intersects(mismatches)) {
								matched++;
								out[i]    = 1u;
								misses[i] = 0u;
							} else if (++misses[i] >= 4u) {
								// Per-draw data behind a relocatable pointer: evaluate it every time.
								flags[i] = 2u;
							}
						}
					};
					select(memory_memo.source_flags, memory_memo.source_mismatches,
					       memory_memo.source_reads, prefill_sources);
					select(memory_memo.flat_flags, memory_memo.flat_mismatches,
					       memory_memo.flat_reads, prefill_flat);
					// Nothing matches any more: the tables changed, record the walk afresh.
					prefill = matched != 0 && (matched * 2 >= candidates);
					if (prefill && memory_memo.active_flag == 1u) {
						if (!memory_memo.active_reads.Intersects(mismatches)) {
							recording_runtime.prefilled_active = memory_memo.active_values;
							memory_memo.active_mismatches      = 0;
						} else if (++memory_memo.active_mismatches >= 4u) {
							memory_memo.active_flag = 2u;
						}
					}
				}
				if (prefill) {
					recording_runtime.prefilled_sources     = prefill_sources;
					recording_runtime.prefilled_values      = memory_memo.source_values;
					recording_runtime.prefilled_flat        = prefill_flat;
					recording_runtime.prefilled_flat_values = memory_memo.flat_values;
				}
				EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
				    entry->second.resource_plan, recording_runtime, resources, specialization));
				recorder.FlagData(data_reads);
				// KYTY_DEBUG_SRT_RELOC=<hex hash>: log the non-relocatable reads of that shader's
				// first full walks, and which recorded read broke the prefill validation.
				static const uint64_t reloc_trace = [] {
					const char* value = std::getenv("KYTY_DEBUG_SRT_RELOC");
					return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
				}();
				if (reloc_trace == params.hash) {
					static uint32_t traced = 0;
					if (traced++ < 3) {
						if (!prefill && memory_memo.valid) {
							for (size_t i = 0; i < memory_memo.reads.size(); i++) {
								const auto& read = memory_memo.reads[i];
								uint32_t    value = 0;
								uint64_t    address = read.address;
								if (read.pair != UINT32_MAX && read.pair + 1u < params.user_data.size()) {
									address = ((static_cast<uint64_t>(params.user_data[read.pair]) |
									            (static_cast<uint64_t>(params.user_data[read.pair + 1]) << 32u)) &
									           0x0000ffffffffffffull) +
									          static_cast<uint64_t>(read.offset);
								}
								const bool ok = read.clean ? ReadShaderGuestMemory(nullptr, address, &value)
								                           : ReadShaderSrtMemory(nullptr, address, &value);
								if (ok != read.ok || (read.data && ok && value != read.value)) {
									LOGF("SrtReloc: prefill broke at read %zu: pair=%u offset=%" PRId64
									     " recorded addr=0x%" PRIx64 " value=%08x now addr=0x%" PRIx64
									     " value=%08x ok=%d/%d clean=%d\n",
									     i, read.pair, read.offset, read.address, read.value, address, value,
									     static_cast<int>(read.ok), static_cast<int>(ok),
									     static_cast<int>(read.clean));
									break;
								}
							}
						}
						for (size_t i = 0; i < recorder.reads.size() && i < 48; i++) {
							const auto& read = recorder.reads[i];
							LOGF("SrtReloc: read %zu pair=%u base=%u offset=%" PRId64 " addr=0x%" PRIx64
							     " value=%08x data=%d clean=%d\n",
							     i, read.pair, read.base_read, read.offset, read.address, read.value,
							     static_cast<int>(read.data), static_cast<int>(read.clean));
						}
						for (size_t i = 0; i < recorder.reads.size(); i++) {
							const auto& read = recorder.reads[i];
							if (read.pair == UINT32_MAX && read.base_read == UINT32_MAX) {
								LOGF("SrtReloc: absolute read %zu addr=0x%" PRIx64 " value=%08x clean=%d\n", i,
								     read.address, read.value, static_cast<int>(read.clean));
							}
						}
					}
				}
				if (memo_stats) {
					auto& st = SrtMemoStats::Map()[params.hash];
					(prefill ? st.prefills : st.prefill_fails)++;
					if (!prefill && !recorder.overflow) {
						st.reads       = recorder.reads.size();
						st.relocatable = static_cast<uint64_t>(std::ranges::count_if(
						    recorder.reads, [](const SrtRead& r) { return r.pair != UINT32_MAX; }));
						st.sources     = source_masks.size();
						st.flagged     = static_cast<uint64_t>(
						    std::ranges::count(source_masks, uint64_t {0}));
					}
					st.overflow += recorder.overflow ? 1u : 0u;
				}
				if (prefill) {
					// The level-1 memo below must revalidate the prefilled bytes as well (with
					// the values just replayed); their base indices move behind this walk's reads.
					const auto shift = static_cast<uint32_t>(recorder.reads.size());
					for (size_t i = 0; i < memory_memo.reads.size(); i++) {
						auto read = memory_memo.reads[i];
						if (read.base_read != UINT32_MAX) {
							read.base_read += shift;
						}
						if (i < replayed.size()) {
							read.value = replayed[i];
						}
						recorder.reads.push_back(read);
					}
				} else if (!recorder.overflow) {
					const auto& plan    = entry->second.resource_plan;
					const auto  sources = plan.descriptor_sources.size();
					memory_memo.valid   = true;
					memory_memo.uses    = 1;
					memory_memo.reads   = recorder.reads;
					memory_memo.source_flags.assign(sources, 0u);
					memory_memo.source_mismatches.assign(sources, 0u);
					memory_memo.source_values.assign(sources, {});
					memory_memo.source_reads.assign(sources, {});
					for (size_t k = 0; k < plan.materialization_sources.size() && k < values.size();
					     k++) {
						const auto source = plan.materialization_sources[k];
						if (source < sources && source < source_masks.size() &&
						    source_masks[source] == 0u && source < source_reads.size() &&
						    !source_reads[source].overflow) {
							memory_memo.source_flags[source]  = 1u;
							memory_memo.source_values[source] = values[k];
							memory_memo.source_reads[source]  = source_reads[source];
						}
					}
					memory_memo.active_flag = active_deps.user == 0 && !active_deps.reads.overflow ? 1u : 0u;
					memory_memo.active_mismatches = 0;
					memory_memo.active_reads      = active_deps.reads;
					memory_memo.active_values     = active_values;
					memory_memo.flat_flags.assign(flat.size(), 0u);
					memory_memo.flat_mismatches.assign(flat.size(), 0u);
					memory_memo.flat_values = flat;
					memory_memo.flat_reads.assign(flat.size(), {});
					for (size_t slot = 0; slot < flat.size() && slot < flat_masks.size(); slot++) {
						if (flat_masks[slot] == 0u && slot < flat_reads.size() &&
						    !flat_reads[slot].overflow) {
							memory_memo.flat_flags[slot] = 1u;
							memory_memo.flat_reads[slot] = flat_reads[slot];
						}
					}
				}
				if (memo_stats) {
					SrtMemoStats::AddWalkTime(
					    params.hash,
					    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                              std::chrono::steady_clock::now() - walk_start)
					                              .count()));
				}
				if ((walk_mask | mask) != mask) {
					mask                        = walk_mask | mask;
					entry->second.user_data_mask = mask;
					memos.clear();
				}
				auto& stored = memos[HashUserData(params.Base(), params.user_data, mask)];
				stored.valid = !recorder.overflow;
				if (stored.valid) {
					stored.shader_base = params.Base();
					stored.user_data.assign(params.user_data.begin(), params.user_data.end());
					stored.reads          = std::move(recorder.reads);
					stored.resources      = resources;
					stored.specialization = specialization;
				}
			}
			KYTY_PROFILER_BLOCK("Programs::Permutation");
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = std::move(resources)};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = params.user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		if (entry == programs.end()) {
			auto resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(resource_plan, runtime, resources,
			                                                    specialization));
			entry = programs.try_emplace(lookup_key, std::move(resource_plan)).first;
		}
		entry->second.permutations.push_back(CompilePermutation(
		    params, options, std::move(translated), std::move(specialization), push_data_cursor));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = std::move(resources)};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	Common::LockGuard lock(m_mutex);
	if (m_driver_cache == nullptr) {
		return;
	}

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	Profiler::ScopedBlock       phase_vertex(KYTY_PROFILER_SOURCE("Programs::PrepareVertex"));
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	phase_vertex.End();
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		KYTY_PROFILER_BLOCK("Programs::PreparePixel");
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	KYTY_PROFILER_BLOCK("Programs::CacheGet");
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	Common::LockGuard lock(m_mutex);
	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[slot]       = bc.color_srcblend;
		static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
		static_params.color_destblend[slot]      = bc.color_destblend;
		static_params.alpha_srcblend[slot]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[slot]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[slot]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
		static_params.blend_enable[slot]         = bc.enable && !rt.info.blend_bypass;
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list =
	    command.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kRectList;
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			EXIT_IF(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX);
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			EXIT_IF(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num));
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				EXIT_IF(index < 0 || index >= vs_input_info.resources_num);
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
		EXIT_IF(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num));
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	Common::LockGuard lock(m_mutex);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
