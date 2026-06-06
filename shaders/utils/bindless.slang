#ifndef BINDLESS
#define BINDLESS
#include "hash.slang"

struct UniformThing {
    float4x4 view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_view_matrix;
    float4x4 inv_projection_matrix;
    float4x4 view_projection_matrix;
    float2 screen_resolution;
    float2 padding;
    float4 position;
    float4 forward;
    float4 sun;
    float4[6] camera_frustum_planes;
    uint32_t debug_type;
    uint32_t toggles_bitmask;
    float32_t time;
};

const static uint SWAPCHAIN_STORAGE_IMAGE_IDX = 0;
const static uint RENDERED_STORAGE_IMAGE_IDX = 1;
const static uint SKYBOX_STORAGE_IMAGE_IDX = 2;
const static uint AMBIENT_SKYBOX_STORAGE_IMAGE_IDX = 3;
const static uint CLOUDS_STORAGE_IMAGE_IDX = 4;
const static uint VOXEL_STORAGE_IMAGE_IDX = 5;

const static uint UNIFORM_BUFFER_THINGY_IDX = 0;
const static uint STORAGE_BUFFER_VERTEX_BUFFER_IDX = 1;
const static uint STORAGE_BUFFER_INDEX_BUFFER_IDX = 2;
const static uint STORAGE_BUFFER_TESS_BUFFER = 3;
const static uint STORAGE_BUFFER_LIGHTS_BUFFER = 4;
const static uint STORAGE_BUFFER_DEBUG_TEXT_LINES = 5;
const static uint STORAGE_BUFFER_VERTEX_COUNTER_IDX = 6;
const static uint STORAGE_BUFFER_INDEX_COUNTER_IDX = 7;


const static uint SKYBOX_SAMPLER_IMAGE_IDX = 0;
const static uint AMBIENT_SKYBOX_SAMPLER_IMAGE_IDX = 1;
const static uint CLOUDS_SAMPLER_IMAGE_IDX = 2;
const static uint ENTIRE_BLOOM_SAMPLED_IMAGE_IDX = 4;

const static uint NEAREST_SAMPLER_IDX = 0;
const static uint LINEAR_SAMPLER_IDX = 1;


[[vk::binding(0, 0)]]
RWTexture2D<float4> storage_images[];

[[vk::binding(0, 0)]]
RWTexture3D<float4> storage_images_3d[];

[[vk::binding(0, 0)]]
RWTexture3D<float> storage_images_3d_float[];

[[vk::binding(0, 0)]]
readonly RWTexture3D<float> readonly_storage_images_3d_float[];

[[vk::binding(0, 0)]]
RWTexture2DArray<float4> storage_images_array[];

[[vk::binding(1, 0)]]
RWByteAddressBuffer storage_buffers[];

[[vk::binding(1, 0)]]
ByteAddressBuffer storage_buffers_readonly[];

[[vk::binding(2, 0)]]
Texture2D<float4> sampled_images[];

[[vk::binding(2, 0)]]
TextureCube<float4> sampled_images_cube[];

[[vk::binding(3, 0)]]
SamplerState samplers[];

[[vk::binding(4, 0)]]
RaytracingAccelerationStructure acceleration_structures[];

UniformThing load_per_frame_uniforms() {
    return storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
}

float hw_trace_shadow(float3 world) {
    bool hit = false;

    RayDesc desc;
    desc.Origin = world;
    desc.Direction = load_per_frame_uniforms().sun.xyz;
    desc.TMax = 100;
    desc.TMin = 0.001;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> sq;
    let flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE;

    sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);

    sq.Proceed();

    return (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

float hw_ao(float3 world, float3 normal) {
    return 1.0;
    
    /*
    RayDesc desc;
    desc.TMax = 10.0;
    desc.TMin = 0.001;
    desc.Origin = world;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> sq;
    let flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE;

    float count = 0.0;

    int samples = 4;

    for (int i = 0; i < samples; i++) {
        desc.Direction = normalize((hash33(world * float3(20.3231, -30.21231, 50.36856) + i * 30.1232) * 2.0 - 1.0) * 0.3 + normal);
        sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);
        sq.Proceed();

        count += sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    }

    return 1 - (count / (float)samples);
    */
}

#endif