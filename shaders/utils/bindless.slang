#ifndef BINDLESS
#define BINDLESS
#include "hash.slang"

public struct UniformThing {
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
    uint32_t padding2;
    uint64_t frame_count;
};

struct GpuModelMetadata {
    uint32_t material_base_index;
    uint32_t storage_buffers_base_index;
};

struct PostProcessPushConstants {
    uint entire_bloom_sampled_image_idx;
};

const static uint SWAPCHAIN_STORAGE_IMAGE_IDX = 0;
const static uint RENDERED_STORAGE_IMAGE_IDX = 1;
const static uint SKYBOX_STORAGE_IMAGE_IDX = 2;
const static uint AMBIENT_SKYBOX_STORAGE_IMAGE_IDX = 3;
const static uint CLOUDS_STORAGE_IMAGE_IDX = 4;
const static uint SDF_TEXTURE_STORAGE_IMAGE_IDX = 5;
const static uint SDF_TEXTURE2_STORAGE_IMAGE_IDX = 6;
const static uint VXGI_STORAGE_IMAGE_IDX = 7;
const static uint CHUNK_LOOKUP_TEXTURE = 8;
const static uint PRE_PASS_TEXTURE = 9;

const static uint UNIFORM_BUFFER_THINGY_IDX = 0;
const static uint STORAGE_BUFFER_DEBUG_TEXT_LINES = 1;

const static uint STORAGE_BUFFER_READBACK_BUFFER = 2;
const static uint READBACK_BUFFER_SDF_CALLS_QUERY_INDEX = 0;
const static uint READBACK_BUFFER_RAYS_TRACED_QUERY_INDEX = 1;
const static uint READBACK_BUFFER_SDF_PRIMITIVE_CALLS_QUERY_INDEX = 2;
const static uint READBACK_BUFFER_SDF_BVH_INNER_LOOP_CALL_QUERY_INDEX = 3;



const static uint STORAGE_BUFFER_MATERIALS_BUFFER = 3;
const static uint STORAGE_BUFFER_SCENE_REPR_BUFFER = 4;
const static uint STORAGE_BUFFER_PRIMITIVE_FLAT_LIST_BUFFER = 5;
const static uint STORAGE_BUFFER_COUNTERS_OF_VARIOUS_TYPES = 6;
const static uint STORAGE_BUFFER_TRANSFORMS_BUFFER = 7;
const static uint STORAGE_BUFFER_CHUNKS_BUFFER = 8;
const static uint STORAGE_BUFFER_BVH_NODES_BUFFER = 9;
const static uint STORAGE_BUFFER_BVH_PRIMITIVE_INDICES_LOOKUP_BUFFER = 10;


const static uint SKYBOX_SAMPLER_IMAGE_IDX = 0;
const static uint AMBIENT_SKYBOX_SAMPLER_IMAGE_IDX = 1;
const static uint CLOUDS_SAMPLER_IMAGE_IDX = 2;
const static uint SDF_TEXTURE_SAMPLED_IMAGE_IDX = 4;
const static uint SDF_TEXTURE2_SAMPLED_IMAGE_IDX = 5;
const static uint VXGI_SAMPLED_IMAGE_IDX = 6;
const static uint SDF_TEXTURE4_SAMPLED_IMAGE_IDX = 7;

const static uint NEAREST_SAMPLER_IDX = 0;
const static uint LINEAR_UNCAMPLED_SAMPLER_IDX = 1;
const static uint LINEAR_CLAMPED_SAMPLER_IDX = 2;


[[vk::binding(0, 0)]]
RWTexture2D<float4> storage_images[];

[[vk::binding(0, 0)]]
RWTexture3D<float4> storage_images_3d[];

[[vk::binding(0, 0)]]
RWTexture3D<float> storage_images_3d_float[];

[[vk::binding(0, 0)]]
RWTexture3D<uint> storage_images_3d_uint[];

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
Texture3D<float4> sampled_images_3d[];

[[vk::binding(2, 0)]]
TextureCube<float4> sampled_images_cube[];

[[vk::binding(3, 0)]]
SamplerState samplers[];

[[vk::binding(4, 0)]]
RaytracingAccelerationStructure acceleration_structures[];

UniformThing load_per_frame_uniforms() {
    // using an aligned load results in higher VGPR and SGPR pressure
    // this leads to overall worse performance by a pretty big margin
    // it (probably) does not help that `UniformThing` is not very well structured 
    return storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
}

public static UniformThing uniforms;

// booleans here packed as bitfield
[SpecializationConstant] const uint SPEC_CONSTANTS_BITFIELD = 0;

// sky spec constants
[SpecializationConstant] const uint SKYBOX_RESOLUTION = 32;
[SpecializationConstant] const uint CLOUDS_RESOLUTION = 32;
[SpecializationConstant] const uint AMBIENT_SKYBOX_RESOLUTION = 32;

// fullscreen pass spec constants
[SpecializationConstant] const uint DOWNSCALE_FACTOR = 1;

// vxgi spec constants
[SpecializationConstant] const uint VXGI_TEXTURE_SIZE = 128;


float4 sample_non_uniform(uint k, SamplerState s, float2 uv) {
    // https://gist.github.com/JuanDiegoMontoya/55482fc04d70e83729bb9528ecdc1c61 
    // nonuniformindex still broken
    // return sampled_images[NonUniformResourceIndex(k)].Sample(s, uv);
    
    while (true) {
        uint idx = WaveReadLaneFirst(k);

        if (idx == k) {
            return sampled_images[NonUniformResourceIndex(idx)].Sample(s, uv);
        }
    }
}


float4 sample_non_uniform_3d(uint k, SamplerState s, float3 uv) {
    // https://gist.github.com/JuanDiegoMontoya/55482fc04d70e83729bb9528ecdc1c61 
    // nonuniformindex still broken
    // return sampled_images_3d[NonUniformResourceIndex(k)].Sample(s, uv);
    
    while (true) {
        uint idx = WaveReadLaneFirst(k);

        if (idx == k) {
            return sampled_images_3d[NonUniformResourceIndex(idx)].Sample(s, uv);
        }
    }
}

#endif