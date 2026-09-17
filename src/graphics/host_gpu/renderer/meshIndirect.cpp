#include "graphics/host_gpu/renderer/meshIndirect.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "gpu_tiler_shaders/mesh_indirect_args_spv.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstring>

namespace Libs::Graphics {

MeshIndirect::MeshIndirect(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_scratch(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, ScratchBytes) {
	SetVulkanObjectNameF(m_graphics.device, m_scratch.Handle(), "Mesh Draw Data");

	vk::PushConstantRange range {};
	range.stageFlags = vk::ShaderStageFlagBits::eCompute;
	range.offset     = 0;
	range.size       = sizeof(PushConstants);
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_layout),
	                     "create mesh indirect pipeline layout");

	const auto                        module = CompileSPV(MESH_INDIRECT_ARGS_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_layout;
	const auto result    = m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info,
	                                                                nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh indirect pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Mesh Indirect Arguments");
}

MeshIndirect::~MeshIndirect() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_layout, nullptr);
}

MeshIndirect::Slot MeshIndirect::Allocate() {
	if (m_cursor + SlotBytes > ScratchBytes) {
		// The previous lap must be consumed before its slots are overwritten.
		KYTY_PROFILER_BLOCK("Wait::MeshIndirectWrap");
		m_scheduler.Wait(m_wrap_tick);
		m_cursor    = 0;
		m_wrap_tick = m_scheduler.CurrentTick();
	}
	if (m_cursor == 0) {
		m_wrap_tick = m_scheduler.CurrentTick();
	}
	Slot slot;
	slot.offset  = m_cursor;
	slot.address = m_scratch.BufferDeviceAddress() + m_cursor;
	m_cursor += SlotBytes;
	return slot;
}

void MeshIndirect::ReadBarrier(vk::CommandBuffer command, const Slot& slot,
                               vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access) {
	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stage;
	barrier.srcAccessMask = src_access;
	barrier.dstStageMask =
	    vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eMeshShaderEXT |
	    vk::PipelineStageFlagBits2::eFragmentShader;
	barrier.dstAccessMask =
	    vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eShaderRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = m_scratch.Handle();
	barrier.offset              = slot.offset;
	barrier.size                = SlotBytes;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

void MeshIndirect::WriteDirect(vk::CommandBuffer command, const Slot& slot,
                               const uint32_t (&draw_data)[DrawDataDwords]) {
	command.updateBuffer(m_scratch.Handle(), slot.offset, sizeof(draw_data), draw_data);
	ReadBarrier(command, slot, vk::PipelineStageFlagBits2::eCopy,
	            vk::AccessFlagBits2::eTransferWrite);
}

void MeshIndirect::Convert(vk::CommandBuffer command, const Slot& slot,
                           const ConvertParams& params) {
	// The guest arguments were written by earlier compute or CP work on this queue.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	PushConstants push {};
	push.arguments            = params.arguments;
	push.output_data          = slot.address;
	push.index_base           = params.index_base;
	push.index_bytes          = params.index_bytes;
	push.primitives_per_group = params.primitives_per_group;
	push.primitive_size       = params.primitive_size;
	push.primitive_step       = params.primitive_step;
	push.max_groups           = params.max_groups;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(1, 1, 1);
	ReadBarrier(command, slot, vk::PipelineStageFlagBits2::eComputeShader,
	            vk::AccessFlagBits2::eShaderWrite);
}

} // namespace Libs::Graphics
