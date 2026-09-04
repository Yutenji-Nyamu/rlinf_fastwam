/*
 * Copyright 2025 Hillbot Inc.
 * Copyright 2020-2024 UCSD SU Lab
 * SPDX-License-Identifier: Apache-2.0
 *
 * Source: sapien-vulkan-2 d8516a4f1467167122ae85f53a8532dbceb1eec2,
 * src/renderer/rt_renderer.cpp, timeline RTRenderer::render overload.
 * The two scene-fence operations below match the existing binary overload.
 * All other rendering operations and denoiser selection are unchanged.
 */
#include <svulkan2/renderer/rt_renderer.h>
#include <stdexcept>

namespace svulkan2::renderer {

// Exact abstract interface from the pinned private denoiser.h. Only this base
// interface is needed; no OIDN/OptiX implementation is rebuilt or substituted.
class Denoiser {
public:
  virtual bool init(bool albedo, bool normal, bool hdr) = 0;
  virtual void allocate(uint32_t width, uint32_t height) = 0;
  virtual void free() = 0;
  virtual void denoise(core::Image &color, core::Image *albedo, core::Image *normal) = 0;
  virtual uint32_t getWidth() const = 0;
  virtual uint32_t getHeight() const = 0;
  virtual ~Denoiser() {};
};

__attribute__((visibility("default")))
void RTRenderer::render(
    scene::Camera &camera,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
    vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const &waitStageMasks,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mScene) {
    throw std::runtime_error("setScene must be called before rendering");
  }

  prepareRender(camera);

  // Reset AFTER prepareRender: its scene-resource updates wait on these fences.
  mContext->getDevice().resetFences(mSceneAccessFence.get());
  mContext->getQueue().submit(mRenderCommandBuffer.get(), {}, {}, {},
                            mSceneAccessFence.get());
#ifdef SVULKAN2_CUDA_INTEROP
  if (mDenoiser) {
    mDenoiser->denoise(mRenderImages.at(mDenoiseColorName)->getImage(),
                      &mRenderImages.at(mDenoiseAlbedoName)->getImage(),
                      &mRenderImages.at(mDenoiseNormalName)->getImage());
  }
#endif
  mContext->getQueue().submit(mPostprocessCommandBuffer.get(), waitSemaphores,
                            waitStageMasks, waitValues, signalSemaphores,
                            signalValues, {});
}
} // namespace svulkan2::renderer
