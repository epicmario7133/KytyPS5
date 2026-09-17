#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <bit>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, uint32_t* value);

// Recorded reads a value depends on, as a bitset over the walk's read indices.
struct SrtReadSet {
	static constexpr uint32_t MaxReads = 512;
	std::array<uint64_t, MaxReads / 64> words {};
	bool                                overflow = false; // a read past MaxReads took part

	void Add(uint32_t index) {
		if (index < MaxReads) {
			words[index / 64u] |= uint64_t {1} << (index % 64u);
		} else {
			overflow = true;
		}
	}
	void Merge(const SrtReadSet& other) {
		for (size_t i = 0; i < words.size(); i++) {
			words[i] |= other.words[i];
		}
		overflow |= other.overflow;
	}
	[[nodiscard]] bool Empty() const {
		if (overflow) {
			return false;
		}
		for (const auto word: words) {
			if (word != 0) {
				return false;
			}
		}
		return true;
	}
	[[nodiscard]] bool Intersects(const SrtReadSet& other) const {
		if (overflow || other.overflow) {
			return true;
		}
		for (size_t i = 0; i < words.size(); i++) {
			if ((words[i] & other.words[i]) != 0) {
				return true;
			}
		}
		return false;
	}
	[[nodiscard]] bool Contains(uint32_t index) const {
		return index >= MaxReads ? overflow
		                         : (words[index / 64u] & (uint64_t {1} << (index % 64u))) != 0;
	}
	// The single read this set holds, if exactly one.
	[[nodiscard]] bool Single(uint32_t& index) const {
		if (overflow) {
			return false;
		}
		int found = -1;
		for (size_t i = 0; i < words.size(); i++) {
			if (words[i] == 0) {
				continue;
			}
			if (found >= 0 || (words[i] & (words[i] - 1)) != 0) {
				return false;
			}
			found = static_cast<int>(i * 64 + std::countr_zero(words[i]));
		}
		if (found < 0) {
			return false;
		}
		index = static_cast<uint32_t>(found);
		return true;
	}
};

// What an evaluated value depends on: user-data dwords (bit 63 stands for registers beyond
// 64) and recorded reads.
struct SrtDeps {
	uint64_t   user = 0;
	SrtReadSet reads;

	void Merge(const SrtDeps& other) {
		user |= other.user;
		reads.Merge(other.reads);
	}
	[[nodiscard]] bool Empty() const { return user == 0 && reads.Empty(); }
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	// When set, receives a bit per user-data dword whose *value* reached a result (or the
	// address of a read that is not relocatable). Pointer pairs used only as the base of
	// relocatable reads are reported through note_read instead.
	uint64_t*                 user_data_mask             = nullptr;
	// Called before every raw memory read with the provenance of its address and returns the
	// read's index: the address is either user-data pair `pair` + `offset`, the 64-bit value
	// of recorded reads (`base_read`, `base_read` + 1) + `offset`, or (both UINT32_MAX) an
	// absolute address.
	uint32_t (*note_read)(void* userdata, uint32_t pair, uint32_t base_read,
	                      int64_t offset) = nullptr;
	// Reads whose values reached a result (a descriptor dword, a flat slot, a branch
	// condition, a non-relocatable address): a memo must compare them, while reads used only
	// as the base of relocatable reads may change freely.
	SrtReadSet* data_reads = nullptr;
	// Optional per-source / per-flat-slot outputs: the user-data dwords whose values each
	// evaluated result depends on (relocatable read addresses excluded).
	std::vector<uint64_t>*   source_masks     = nullptr;
	std::vector<uint64_t>*   flat_masks       = nullptr;
	std::vector<SrtReadSet>* source_read_sets = nullptr;
	std::vector<SrtReadSet>* flat_read_sets   = nullptr;
	// Optional prefilled results (indexed by descriptor source / flat slot): a source or slot
	// whose flag is set is copied instead of evaluated.
	// Optional copies of the evaluated results (by position in the evaluated source list, and
	// by flat slot) so a caller can build such prefills.
	std::vector<DescriptorValue>* evaluated_values = nullptr;
	std::vector<uint32_t>*        evaluated_flat   = nullptr;
	// The control-flow result (active sources) and what it depended on, for prefilling.
	SrtDeps*                 active_deps      = nullptr;
	std::vector<uint8_t>*    evaluated_active = nullptr;
	std::span<const uint8_t> prefilled_active;
	std::span<const uint8_t>         prefilled_sources;
	std::span<const DescriptorValue> prefilled_values;
	std::span<const uint8_t>         prefilled_flat;
	std::span<const uint32_t>        prefilled_flat_values;
};

enum class RuntimeValueType { Any, Integer };

// Collects reachable ReadConst values. Immediate offsets receive compact flat-buffer slots;
// dynamic offsets remain explicit and are never assigned a fake slot.
void BuildSrtPlan(Program& program);
bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
bool EvaluateUniformValues(const ResourcePlan& program, std::span<const Value> values,
                            const SrtRuntime& runtime, std::span<uint32_t> results);

bool EvaluateDescriptorSource(const ResourcePlan& program, uint32_t source,
                              const SrtRuntime& runtime, DescriptorValue& result);

// Evaluates one runtime snapshot transactionally. Scalar values and ReadConst results shared by
// several descriptors are memoized once across the batch.
bool EvaluateDescriptorSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                               const SrtRuntime& runtime, std::vector<DescriptorValue>& results);

// Evaluates potentially reachable descriptor sources and the flattened immediate SRT with one
// memoized scalar walk. Inactive descriptors are zero; on failure no destination is changed.
bool EvaluateRuntimeSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                            const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                            std::vector<uint32_t>& flat, std::span<const uint8_t> clean_flat_slots,
                            std::vector<uint8_t>& active_sources);

bool WalkSrt(const ResourcePlan& program, const SrtRuntime& runtime,
             std::vector<uint32_t>& flat);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
