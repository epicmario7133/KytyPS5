#include "graphics/host_gpu/renderer/cache/asyncCopyPool.h"

#include "common/profiler.h"
#include "kernel/memory.h"

#include <cstring>

namespace Libs::Graphics {

AsyncCopyPool::AsyncCopyPool() {
	constexpr unsigned Workers = 2;
	for (unsigned index = 0; index < Workers; index++) {
		m_workers.emplace_back([this] { Run(); });
	}
}

AsyncCopyPool::~AsyncCopyPool() {
	Join();
	{
		std::lock_guard lock(m_mutex);
		m_stopping = true;
	}
	m_work_available.notify_all();
	for (auto& worker: m_workers) {
		worker.join();
	}
}

void AsyncCopyPool::Enqueue(const Job& job) {
	{
		std::lock_guard lock(m_mutex);
		m_queue.push_back(job);
		m_pending.fetch_add(1, std::memory_order_relaxed);
	}
	m_work_available.notify_one();
}

void AsyncCopyPool::Join() {
	if (m_pending.load(std::memory_order_acquire) == 0) {
		return;
	}
	KYTY_PROFILER_BLOCK("Wait::AsyncCopies");
	std::unique_lock lock(m_mutex);
	m_work_done.wait(lock, [this] { return m_pending.load(std::memory_order_acquire) == 0; });
}

void AsyncCopyPool::Run() {
	for (;;) {
		Job job;
		{
			std::unique_lock lock(m_mutex);
			m_work_available.wait(lock, [this] { return m_stopping || !m_queue.empty(); });
			if (m_queue.empty()) {
				return;
			}
			job = m_queue.back();
			m_queue.pop_back();
		}
		// A range the guest unmapped in the meantime reads as zero; the image is refreshed
		// again once the guest writes it.
		if (!LibKernel::Memory::TryReadBacking(job.vaddr, job.destination, job.size) &&
		    !LibKernel::Memory::TryReadPrtBacking(job.vaddr, job.destination, job.size)) {
			std::memset(job.destination, 0, job.size);
		}
		{
			std::lock_guard lock(m_mutex);
			if (m_pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
				m_work_done.notify_all();
			}
		}
	}
}

} // namespace Libs::Graphics
