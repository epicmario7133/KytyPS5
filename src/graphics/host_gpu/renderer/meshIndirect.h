#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandScheduler;

// Draw data for the host mesh-shader path. Mesh programs read their draw parameters (vertex
// count, offsets, index pointer) from a device buffer whose address travels in push constants,
// so GPU-written DRAW_INDIRECT arguments can be converted on the GPU instead of being read back
// on the CPU, which drains the queue every frame.
class MeshIndirect final {
public:
	// Draw-data dwords followed by the VkDrawMeshTasksIndirectCommandEXT.
	static constexpr uint32_t DrawDataDwords = 6;
	static constexpr uint64_t TasksOffset    = DrawDataDwords * sizeof(uint32_t);
	static constexpr uint64_t SlotBytes      = 64;

	struct Slot {
		uint64_t          offset  = 0;
		vk::DeviceAddress address = 0;
	};

	struct ConvertParams {
		vk::DeviceAddress arguments           = 0; // guest arguments (GPU-written)
		vk::DeviceAddress index_base          = 0;
		uint32_t          index_bytes         = 0; // 0 for non-indexed draws
		uint32_t          primitives_per_group = 0;
		uint32_t          primitive_size      = 0;
		uint32_t          primitive_step      = 0;
		uint32_t          max_groups          = 0;
	};

	MeshIndirect(GraphicContext& graphics, CommandScheduler& scheduler);
	~MeshIndirect();
	KYTY_CLASS_NO_COPY(MeshIndirect);

	[[nodiscard]] Slot Allocate();
	// Host-provided draw data; recorded outside a render pass.
	void WriteDirect(vk::CommandBuffer command, const Slot& slot,
	                 const uint32_t (&draw_data)[DrawDataDwords]);
	// GPU conversion of guest indirect arguments; recorded outside a render pass.
	void Convert(vk::CommandBuffer command, const Slot& slot, const ConvertParams& params);

	[[nodiscard]] vk::Buffer Handle() const noexcept { return m_scratch.Handle(); }

private:
	struct PushConstants {
		uint64_t arguments;
		uint64_t output_data;
		uint64_t index_base;
		uint32_t index_bytes;
		uint32_t primitives_per_group;
		uint32_t primitive_size;
		uint32_t primitive_step;
		uint32_t max_groups;
		uint32_t padding;
	};
	static constexpr uint64_t ScratchBytes = 1024 * 1024;

	void ReadBarrier(vk::CommandBuffer command, const Slot& slot, vk::PipelineStageFlags2 src_stage,
	                 vk::AccessFlags2 src_access);

	GraphicContext&    m_graphics;
	CommandScheduler&  m_scheduler;
	Buffer             m_scratch;
	uint64_t           m_cursor    = 0;
	uint64_t           m_wrap_tick = 0;
	vk::PipelineLayout m_layout;
	vk::Pipeline       m_pipeline;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
