#ifndef BINDLESS
#define BINDLESS

const static uint SWAPCHAIN_STORAGE_IMAGE_IDX = 0;
const static uint RENDERED_STORAGE_IMAGE_IDX = 1;
const static uint SKYBOX_STORAGE_IMAGE_IDX = 2;
const static uint CLOUDS_STORAGE_IMAGE_IDX = 3;

const static uint UNIFORM_BUFFER_THINGY_IDX = 0;
const static uint STORAGE_BUFFER_VERTEX_BUFFER_IDX = 1;
const static uint STORAGE_BUFFER_INDEX_BUFFER = 2;
const static uint STORAGE_BUFFER_TESS_BUFFER = 3;

const static uint SKYBOX_SAMPLER_IMAGE_IDX = 0;
const static uint CLOUDS_SAMPLER_IMAGE_IDX = 1;
const static uint ENTIRE_BLOOM_SAMPLED_IMAGE_IDX = 3;

[[vk::binding(0, 0)]]
RWTexture2D<float4> storage_images[];

[[vk::binding(0, 0)]]
RWTexture2DArray<float4> storage_images_array[];

[[vk::binding(1, 0)]]
RWByteAddressBuffer storage_buffers[];

[[vk::binding(1, 0)]]
ByteAddressBuffer storage_buffers_readonly[];

[[vk::binding(2, 0)]]
Sampler2D<float4> sampled_images[];

[[vk::binding(2, 0)]]
SamplerCube<float4> sampled_images_cube[];

#endif