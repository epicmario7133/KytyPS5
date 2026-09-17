#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_image_pool(graphics),
      m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// KYTY_DEBUG_FAULT_TRACE=1: count guest page faults; log the GPU thread's read faults.
	static const bool fault_trace = std::getenv("KYTY_DEBUG_FAULT_TRACE") != nullptr;
	if (fault_trace) {
		const bool gpu_thread = GuestGpu::IsGpuThread();
		m_fault_count.fetch_add(1, std::memory_order_relaxed);
		if (gpu_thread) {
			const auto n = m_gpu_fault_count.fetch_add(1, std::memory_order_relaxed);
			if (n < 256 || (n % 4096) == 0) {
				LOGF("FaultTrace[%" PRIu64 "]: gpu-thread %s fault at 0x%016" PRIx64 "\n", n,
				     access == PageFaultAccess::Write ? "write" : "read", fault_vaddr);
			}
		}
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		if (m_command_scheduler.Active()) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
		m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
	});
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();

	// KYTY_DEBUG_MEM_STATS=1: log device memory use and cache population every 256 runs.
	static const bool mem_stats = std::getenv("KYTY_DEBUG_MEM_STATS") != nullptr;
	static uint32_t   mem_stats_tick = 0;
	if (mem_stats && (mem_stats_tick++ % 256) == 0) {
		const auto [image_bytes, image_count]   = m_texture_cache.MemoryStats();
		const auto [buffer_bytes, buffer_count] = m_buffer_cache.MemoryStats();
		const auto& master                      = m_command_scheduler.GetMasterSemaphore();
		LOGF("MemStats: frame=%d device=%" PRIu64 " MiB budget=%" PRIu64 " MiB images=%zu (%" PRIu64
		     " MiB) buffers=%zu (%" PRIu64 " MiB) ticks=%" PRIu64 " waits=%" PRIu64 " (%" PRIu64
		     " ms) drains=%" PRIu64 " stream_waits=%" PRIu64 " readbacks=%" PRIu64
		     " downloads=%" PRIu64 " faults=%" PRIu64 " gpu_faults=%" PRIu64
		     " stale_pages=%" PRIu64 "\n",
		     m_gpu != nullptr ? m_gpu->GetFrameNum() : -1, m_graphics.GetDeviceMemoryUsage() >> 20,
		     m_graphics.GetTotalMemoryBudget() >> 20, image_count, image_bytes >> 20, buffer_count,
		     buffer_bytes >> 20, m_command_scheduler.CurrentTick(), master.BlockingWaits(),
		     master.BlockingWaitNanoseconds() / 1000000, m_command_scheduler.DrainCount(),
		     m_command_scheduler.StreamWaitCount(), m_buffer_cache.ReadbackCount(),
		     m_texture_cache.DownloadCount(), m_fault_count.load(std::memory_order_relaxed),
		     m_gpu_fault_count.load(std::memory_order_relaxed), m_buffer_cache.StalePageCount());
		if (m_gpu != nullptr) {
			const auto stats = m_gpu->GetThreadStats();
			LOGF("GpuThread: idle=%" PRIu64 " ms blocked=%" PRIu64 " ms (%" PRIu64
			     " polls) process=%" PRIu64 " ms commands=%" PRIu64 " ms\n",
			     stats.idle_ns / 1000000, stats.blocked_ns / 1000000, stats.blocked,
			     stats.process_ns / 1000000, stats.command_ns / 1000000);
		}
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
