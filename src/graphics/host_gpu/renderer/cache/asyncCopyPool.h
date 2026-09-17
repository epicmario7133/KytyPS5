#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_ASYNCCOPYPOOL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_ASYNCCOPYPOOL_H_

#include "common/common.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// Copies guest memory into host-visible staging on worker threads so the command processor
// does not spend its time in memcpy for large texture uploads. Every job must be joined
// before the command buffer that reads the staging bytes is submitted.
class AsyncCopyPool final {
public:
	struct Job {
		uint64_t vaddr       = 0;
		uint8_t* destination = nullptr;
		uint64_t size        = 0;
	};

	AsyncCopyPool();
	~AsyncCopyPool();
	KYTY_CLASS_NO_COPY(AsyncCopyPool);

	void Enqueue(const Job& job);
	// Blocks until every enqueued copy has landed.
	void Join();
	[[nodiscard]] bool Pending() const noexcept { return m_pending.load() != 0; }

private:
	void Run();

	std::vector<std::thread> m_workers;
	std::mutex               m_mutex;
	std::condition_variable  m_work_available;
	std::condition_variable  m_work_done;
	std::vector<Job>         m_queue;
	std::atomic<uint32_t>    m_pending {0};
	bool                     m_stopping = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_ASYNCCOPYPOOL_H_
