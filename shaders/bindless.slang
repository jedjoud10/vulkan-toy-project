#ifndef BINDLESS
#define BINDLESS

[[vk::binding(0, 0)]]
RWTexture2D<float4>[] storage_images;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float4>[] storage_buffers;

[[vk::binding(2, 0)]]
Sampler2D<float4>[] sampled_images;


#endif