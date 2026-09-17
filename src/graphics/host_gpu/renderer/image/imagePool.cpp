#include "graphics/host_gpu/renderer/image/imagePool.h"

#include "common/assert.h"

#include <utility>

namespace Libs::Graphics {

namespace {

void MoveBacking(VulkanImage& from, VulkanImage& to) {
	to.format             = from.format;
	to.image_type         = from.image_type;
	to.extent             = from.extent;
	to.layers             = from.layers;
	to.mip_levels         = from.mip_levels;
	to.samples            = from.samples;
	to.usage              = from.usage;
	to.flags              = from.flags;
	to.image              = std::exchange(from.image, nullptr);
	to.allocation         = std::exchange(from.allocation, nullptr);
	to.state              = from.state;
	to.subresource_states = std::move(from.subresource_states);
}

bool Matches(const VulkanImage& backing, const vk::ImageCreateInfo& create) {
	return backing.format == create.format && backing.image_type == create.imageType &&
	       backing.extent == create.extent && backing.mip_levels == create.mipLevels &&
	       backing.layers == create.arrayLayers &&
	       backing.samples == static_cast<uint32_t>(create.samples) &&
	       backing.usage == create.usage && backing.flags == create.flags;
}

} // namespace

ImagePool::~ImagePool() {
	Clear();
}

bool ImagePool::Take(const vk::ImageCreateInfo& create, VulkanImage& backing,
                     std::vector<CachedImageView>& views) {
	for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
		auto& entry = **it;
		if (!Matches(entry.backing, create)) {
			continue;
		}
		MoveBacking(entry.backing, backing);
		views = std::move(entry.views);
		m_bytes -= entry.bytes;
		m_entries.erase(it);
		return true;
	}
	return false;
}

void ImagePool::Offer(VulkanImage& backing, std::vector<CachedImageView>& views,
                      uint64_t bytes) {
	EXIT_IF(backing.image == nullptr);
	auto entry = std::make_unique<Entry>();
	MoveBacking(backing, entry->backing);
	entry->views = std::move(views);
	entry->bytes = bytes;
	views.clear();
	m_bytes += bytes;
	m_entries.push_back(std::move(entry));
	while (!m_entries.empty() && (m_entries.size() > MaxEntries || m_bytes > MaxBytes)) {
		Destroy(*m_entries.front());
		m_entries.pop_front();
	}
}

void ImagePool::Clear() {
	for (auto& entry: m_entries) {
		Destroy(*entry);
	}
	m_entries.clear();
}

void ImagePool::Destroy(Entry& entry) {
	for (const auto& cached: entry.views) {
		if (cached.view != nullptr) {
			m_graphics.device.destroyImageView(cached.view, nullptr);
		}
	}
	entry.views.clear();
	if (entry.backing.image != nullptr) {
		m_graphics.DeleteImage(entry.backing);
	}
	m_bytes -= entry.bytes;
	entry.bytes = 0;
}

} // namespace Libs::Graphics
