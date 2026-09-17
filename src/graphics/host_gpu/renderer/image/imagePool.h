#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEPOOL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEPOOL_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace Libs::Graphics {

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

// Retired Vulkan images kept for reuse. Titles that alias transient render targets (a color
// buffer, a depth buffer and another color format at the same guest address every frame) make
// the texture cache replace its images dozens of times per frame; recreating the image, its
// memory and its views each time costs more than the draws that use them.
class ImagePool final {
public:
	explicit ImagePool(GraphicContext& graphics): m_graphics(graphics) {}
	~ImagePool();
	KYTY_CLASS_NO_COPY(ImagePool);

	// Takes a retired image with exactly this creation state, or returns false.
	[[nodiscard]] bool Take(const vk::ImageCreateInfo& create, VulkanImage& backing,
	                        std::vector<CachedImageView>& views);
	// Parks a retired image (the GPU must be done with it); destroys the oldest when full.
	void Offer(VulkanImage& backing, std::vector<CachedImageView>& views, uint64_t bytes);
	// Releases everything (memory pressure).
	void Clear();

	[[nodiscard]] uint64_t Bytes() const noexcept { return m_bytes; }
	[[nodiscard]] size_t   Count() const noexcept { return m_entries.size(); }

private:
	struct Entry {
		VulkanImage                  backing;
		std::vector<CachedImageView> views;
		uint64_t                     bytes = 0;
	};
	static constexpr size_t   MaxEntries = 96;
	static constexpr uint64_t MaxBytes   = 384ull * 1024 * 1024;

	void Destroy(Entry& entry);

	GraphicContext&   m_graphics;
	std::deque<std::unique_ptr<Entry>> m_entries;
	uint64_t          m_bytes = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEPOOL_H_
