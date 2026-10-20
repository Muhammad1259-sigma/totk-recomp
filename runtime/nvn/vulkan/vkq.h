// Deferred command recording (2026-10-17, Deck: the render thread is the frame limit at ~50 fps standing, 90-95 % of a
// core; d34_perf: 29 % of its samples were in libvulkan_radeon, mostly command emission). With LARECOMP_VKQ=1 every
// vkCmd* / vkBegin/EndCommandBuffer call below is packed into a ring and replayed in order by one "vk-record" thread, as
// DXVK's CS thread does; the caller only copies the arguments. vk_submit_extra drains the ring before vkQueueSubmit2.
// Function-like macros: "(vkCmdDraw)(...)" (parenthesised name) still calls the real function (vkq_impl.h does).
#pragma once
#include <vulkan/vulkan.h>

void vkq_drain(void);   // wait until the record thread has replayed everything queued so far (no-op when off)
int vkq_enabled(void);

VkResult q_vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo *);
VkResult q_vkEndCommandBuffer(VkCommandBuffer);
void q_vkCmdDraw(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t);
void q_vkCmdDrawIndexed(VkCommandBuffer, uint32_t, uint32_t, uint32_t, int32_t, uint32_t);
void q_vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
void q_vkCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
void q_vkCmdBindDescriptorSets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet *,
                               uint32_t, const uint32_t *);
void q_vkCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t, uint32_t, const void *);
void q_vkCmdBindVertexBuffers(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer *, const VkDeviceSize *);
void q_vkCmdBindIndexBuffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType);
void q_vkCmdSetViewport(VkCommandBuffer, uint32_t, uint32_t, const VkViewport *);
void q_vkCmdSetScissor(VkCommandBuffer, uint32_t, uint32_t, const VkRect2D *);
void q_vkCmdSetBlendConstants(VkCommandBuffer, const float[4]);
void q_vkCmdSetDepthBias(VkCommandBuffer, float, float, float);
void q_vkCmdSetDepthBiasEnable(VkCommandBuffer, VkBool32);
void q_vkCmdSetDepthTestEnable(VkCommandBuffer, VkBool32);
void q_vkCmdSetDepthWriteEnable(VkCommandBuffer, VkBool32);
void q_vkCmdSetDepthCompareOp(VkCommandBuffer, VkCompareOp);
void q_vkCmdSetStencilTestEnable(VkCommandBuffer, VkBool32);
void q_vkCmdSetStencilOp(VkCommandBuffer, VkStencilFaceFlags, VkStencilOp, VkStencilOp, VkStencilOp, VkCompareOp);
void q_vkCmdSetStencilCompareMask(VkCommandBuffer, VkStencilFaceFlags, uint32_t);
void q_vkCmdSetStencilWriteMask(VkCommandBuffer, VkStencilFaceFlags, uint32_t);
void q_vkCmdSetStencilReference(VkCommandBuffer, VkStencilFaceFlags, uint32_t);
void q_vkCmdSetCullMode(VkCommandBuffer, VkCullModeFlags);
void q_vkCmdSetFrontFace(VkCommandBuffer, VkFrontFace);
void q_vkCmdSetPrimitiveTopology(VkCommandBuffer, VkPrimitiveTopology);
void q_vkCmdSetPrimitiveRestartEnable(VkCommandBuffer, VkBool32);
void q_vkCmdBeginRendering(VkCommandBuffer, const VkRenderingInfo *);
void q_vkCmdEndRendering(VkCommandBuffer);
void q_vkCmdPipelineBarrier2(VkCommandBuffer, const VkDependencyInfo *);
void q_vkCmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy *);
void q_vkCmdCopyImage(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageCopy *);
void q_vkCmdCopyBufferToImage(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, uint32_t, const VkBufferImageCopy *);
void q_vkCmdCopyImageToBuffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, uint32_t, const VkBufferImageCopy *);
void q_vkCmdBlitImage(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageBlit *, VkFilter);
void q_vkCmdResolveImage(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageResolve *);
void q_vkCmdClearColorImage(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue *, uint32_t, const VkImageSubresourceRange *);
void q_vkCmdClearDepthStencilImage(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue *, uint32_t,
                                   const VkImageSubresourceRange *);
void q_vkCmdClearAttachments(VkCommandBuffer, uint32_t, const VkClearAttachment *, uint32_t, const VkClearRect *);
void q_vkCmdResetQueryPool(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t);
void q_vkCmdWriteTimestamp2(VkCommandBuffer, VkPipelineStageFlags2, VkQueryPool, uint32_t);
// vkCmdSetAttachmentFeedbackLoopEnableEXT through its function pointer (queued in order like the rest)
void q_set_fbl(PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT fn, VkCommandBuffer cb, VkImageAspectFlags m);

#define vkBeginCommandBuffer(...) q_vkBeginCommandBuffer(__VA_ARGS__)
#define vkEndCommandBuffer(...) q_vkEndCommandBuffer(__VA_ARGS__)
#define vkCmdDraw(...) q_vkCmdDraw(__VA_ARGS__)
#define vkCmdDrawIndexed(...) q_vkCmdDrawIndexed(__VA_ARGS__)
#define vkCmdDispatch(...) q_vkCmdDispatch(__VA_ARGS__)
#define vkCmdBindPipeline(...) q_vkCmdBindPipeline(__VA_ARGS__)
#define vkCmdBindDescriptorSets(...) q_vkCmdBindDescriptorSets(__VA_ARGS__)
#define vkCmdPushConstants(...) q_vkCmdPushConstants(__VA_ARGS__)
#define vkCmdBindVertexBuffers(...) q_vkCmdBindVertexBuffers(__VA_ARGS__)
#define vkCmdBindIndexBuffer(...) q_vkCmdBindIndexBuffer(__VA_ARGS__)
#define vkCmdSetViewport(...) q_vkCmdSetViewport(__VA_ARGS__)
#define vkCmdSetScissor(...) q_vkCmdSetScissor(__VA_ARGS__)
#define vkCmdSetBlendConstants(...) q_vkCmdSetBlendConstants(__VA_ARGS__)
#define vkCmdSetDepthBias(...) q_vkCmdSetDepthBias(__VA_ARGS__)
#define vkCmdSetDepthBiasEnable(...) q_vkCmdSetDepthBiasEnable(__VA_ARGS__)
#define vkCmdSetDepthTestEnable(...) q_vkCmdSetDepthTestEnable(__VA_ARGS__)
#define vkCmdSetDepthWriteEnable(...) q_vkCmdSetDepthWriteEnable(__VA_ARGS__)
#define vkCmdSetDepthCompareOp(...) q_vkCmdSetDepthCompareOp(__VA_ARGS__)
#define vkCmdSetStencilTestEnable(...) q_vkCmdSetStencilTestEnable(__VA_ARGS__)
#define vkCmdSetStencilOp(...) q_vkCmdSetStencilOp(__VA_ARGS__)
#define vkCmdSetStencilCompareMask(...) q_vkCmdSetStencilCompareMask(__VA_ARGS__)
#define vkCmdSetStencilWriteMask(...) q_vkCmdSetStencilWriteMask(__VA_ARGS__)
#define vkCmdSetStencilReference(...) q_vkCmdSetStencilReference(__VA_ARGS__)
#define vkCmdSetCullMode(...) q_vkCmdSetCullMode(__VA_ARGS__)
#define vkCmdSetFrontFace(...) q_vkCmdSetFrontFace(__VA_ARGS__)
#define vkCmdSetPrimitiveTopology(...) q_vkCmdSetPrimitiveTopology(__VA_ARGS__)
#define vkCmdSetPrimitiveRestartEnable(...) q_vkCmdSetPrimitiveRestartEnable(__VA_ARGS__)
#define vkCmdBeginRendering(...) q_vkCmdBeginRendering(__VA_ARGS__)
#define vkCmdEndRendering(...) q_vkCmdEndRendering(__VA_ARGS__)
#define vkCmdPipelineBarrier2(...) q_vkCmdPipelineBarrier2(__VA_ARGS__)
#define vkCmdCopyBuffer(...) q_vkCmdCopyBuffer(__VA_ARGS__)
#define vkCmdCopyImage(...) q_vkCmdCopyImage(__VA_ARGS__)
#define vkCmdCopyBufferToImage(...) q_vkCmdCopyBufferToImage(__VA_ARGS__)
#define vkCmdCopyImageToBuffer(...) q_vkCmdCopyImageToBuffer(__VA_ARGS__)
#define vkCmdBlitImage(...) q_vkCmdBlitImage(__VA_ARGS__)
#define vkCmdResolveImage(...) q_vkCmdResolveImage(__VA_ARGS__)
#define vkCmdClearColorImage(...) q_vkCmdClearColorImage(__VA_ARGS__)
#define vkCmdClearDepthStencilImage(...) q_vkCmdClearDepthStencilImage(__VA_ARGS__)
#define vkCmdClearAttachments(...) q_vkCmdClearAttachments(__VA_ARGS__)
#define vkCmdResetQueryPool(...) q_vkCmdResetQueryPool(__VA_ARGS__)
#define vkCmdWriteTimestamp2(...) q_vkCmdWriteTimestamp2(__VA_ARGS__)
