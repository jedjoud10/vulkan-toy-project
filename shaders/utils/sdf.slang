#ifndef SDF
#define SDF

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"
#include "sdf_iq.slang"
#include "sdf_ext.slang"


const static float CHUNK_PHYSICAL_SIZE = 4.0;
const static float CHUNK_LOGICAL_SIZE = 64;

const static uint CHUNK_LOOKUP_MEGA_TEXTURE_SIZE = 128;
const static uint CHUNK_LOOKUP_MEGA_TEXTURE_HALF_SIZE = CHUNK_LOOKUP_MEGA_TEXTURE_SIZE / 2;

const static uint MAX_ITERS = 256;
const static float MAX_DISTANCE = 10000;
const static float THRESHOLD_DISTANCE_PREPASS = 0.01;
const static float THRESHOLD_DISTANCE = 0.001;
const static float DISTANCE_SCALING_FACTOR = 0.9;

const static uint DDA_TRAVERSAL_CHUNKS_ITER = 32;
const static uint DDA_TRAVERSAL_RAY_MARCH_ITER = 128;
const static float CHUNK_BOUNDARY_OFFSET_EPS = 1e-5;

// #define SHADOWS 

struct PackedBvhNode {
    float16_t3 min;
    uint16_t prim_count;
    float16_t3 max;
    uint16_t first_index;
};

struct PrimitiveNode {
    uint transform_index;
    uint type;
}

enum SdfType {
    Box,
    BoxWithCutout,
    Rock,
    Interesting1,
    Tree,
}

float get_sdf_from_type(float3 local, SdfType type, uint object_index) {

    switch (type) {
        case SdfType.Box:
            return sdBox(local, 0.5);
        case SdfType.BoxWithCutout:
            float box = sdBox(local, 0.99);
            box = max(box, -sdCylinder(local.xyz, float3(0.0, 0.0, 0.6)));
            box = max(box, -sdCylinder(local.xzy, float3(0.0, 0.0, 0.6)));
            box = max(box, -sdCylinder(local.yxz, float3(0.0, 0.0, 0.6)));
            
            box = max(box, sdSphere(local, 1.0));            
            return box;
        case SdfType.Rock:
            float s = sdBox(local, 0.8);
            float blending = noise(local * 10) * 0.2 + 0.1;

            [ForceUnroll]
            for (int i = 0; i < 4; i++) {
                int seed = i + object_index * 52;
                float3 normal = normalize(float3(sin(seed * 40.21), sin(seed * 10.21), sin(seed * -20.43)));
                if (length(normal) > 0.01) {
                    float k = dot(normalize(normal), local) - 0.5;
                    s = smax(s, k, blending);
                }
            }

            return s;
        case SdfType.Interesting1:
            s = max(sdBox(local, 0.8), sdCylinder(local, float3(0, 0, 0.4)));

            float3 k = mul(local, AngleAxis3x3(uniforms.time, normalize(hash31(object_index))));
            k.x = abs(k.x);
            k = k - float3(0.6, 0.4, 0.0);
            s = min(sdSphere(k, 0.2), s);

            s = min(max(sdSphere(local - float3(0, -0.3, 0), 0.6), sdSphere(local - float3(0, 0.3, 0), 0.6)), s);

            s += sin(local.y*30 + uniforms.time*4.0)*0.01;

            return s;

        case SdfType.Tree:
            return tree2(local*8, object_index).x/8;
    }

    return 0.0;
}

float3 compute_normals(float3 p) {
    float eps = 0.001;

    float x = d(p - float3(eps, 0, 0)).x;
    float y = d(p - float3(0, eps, 0)).x;
    float z = d(p - float3(0, 0, eps)).x;
    float b = d(p).x;
    return normalize(float3(b-x, b-y, b-z));
    

    /*
    // these "higher-quality" normals don't look any different lmfao
    float nx = d(p - float3(eps, 0, 0)).x;
    float ny = d(p - float3(0, eps, 0)).x;
    float nz = d(p - float3(0, 0, eps)).x;
    float px = d(p + float3(eps, 0, 0)).x;
    float py = d(p + float3(0, eps, 0)).x;
    float pz = d(p + float3(0, 0, eps)).x;
    return normalize(float3(px-nx, py-ny, pz-nz));
    */
}

float4 sample_chunk(float3 p, uint chunk_start_index) {
    uint target_texture_index = storage_buffers_readonly[STORAGE_BUFFER_CHUNKS_BUFFER].Load<uint>(chunk_start_index * 4);
    float3 local_position = p - floor(p / CHUNK_PHYSICAL_SIZE) * CHUNK_PHYSICAL_SIZE;

    // FIXME: there is a problem here with bilinear filtering. issue happens on the positive axis, near the positive edge of the chunk. seems like we need a 1 pixel margin?
    float4 sampled = sample_non_uniform_3d(target_texture_index, samplers[LINEAR_CLAMPED_SAMPLER_IDX], (local_position) / CHUNK_PHYSICAL_SIZE);
    return sampled;
}

void apply_primitive_sdf(float3 p, PrimitiveNode node, int node_index, inout float sdf) {
    // 3x4 = 3 rows, 4 columns
    // matrix must be the INVERSE of the transform matrix
    uint transform_byte_offset = node.transform_index * 4 * 4 * 3;
    float3x4 transform = storage_buffers_readonly[STORAGE_BUFFER_TRANSFORMS_BUFFER].Load<float3x4>(transform_byte_offset);

    // 3x4 * 4x1 = 3x1 vector
    float3 local = mul(transform, float4(p, 1));

    SdfType sdf_type = SdfType(node.type);
    // sdf_type = SdfType.Tree;
    
    float obj_sdf = get_sdf_from_type(local.xyz, sdf_type, node_index); 

    // apply the object SDF to the global sdf
    // TODO: separate the BVH primitives into "layers" that each have their own separate merging rules (between their objects *and* between layers)
    // would allow us to compose cool terrain stuff like dat
    sdf = smin(sdf, obj_sdf, 0.1);
    // sdf = min(sdf, obj_sdf);
    // sdf = max(sdf, -obj_sdf);
}

// TODO: optimize BVH traversal
float d(float3 p) {
    capture_performance_sdf_call();


    float sdf = p.y;
            
    uint len = 0;
    const static uint capacity = 10;
    uint[capacity] arrays = { 0 };

    len += 1;
    
    while (len > 0) {
        capture_performance_bvh_distance_query_inner_loop();
        len -= 1;
        uint index = arrays[len];
        PackedBvhNode node = storage_buffers_readonly[STORAGE_BUFFER_BVH_NODES_BUFFER].Load<PackedBvhNode>(sizeof(PackedBvhNode) * index);

        float3 aabb_min = node.min.xyz;
        float3 aabb_max = node.max.xyz;
        float3 center = (aabb_min+aabb_max)*0.5;
        float3 half_extent = (aabb_max-aabb_min) * 0.5;
        float bvh_node_sdf = sdBox(p-center, half_extent);

        if (node.prim_count == 0) {
            // NOTE: this is the check causes subtractive SDFs not to work properly
            if (sdf > bvh_node_sdf) {
                if (len < capacity) {
                    PackedBvhNode first_child = storage_buffers_readonly[STORAGE_BUFFER_BVH_NODES_BUFFER].Load<PackedBvhNode>(sizeof(PackedBvhNode) * node.first_index);

                    // make sure that we visits the BVH node that the point is contained in first
                    // it is more likely that the surface is there, probably
                    if (all(p > first_child.min.xyz & p < first_child.max.xyz)) {
                        arrays[len] = node.first_index;
                        arrays[len+1] = node.first_index+1;
                    } else {
                        arrays[len] = node.first_index+1;
                        arrays[len+1] = node.first_index;
                    }

                    len += 2;
                }
            }
        } else {
            // NOTE: this is another check causes that also subtractive SDFs not to work properly
            // we prune too aggressively it seems
            if (sdf < bvh_node_sdf) {
                continue;
            }

            for (int i = 0; i < node.prim_count; i++) {
                capture_performance_sdf_primitive_call();
                uint node_index = storage_buffers_readonly[STORAGE_BUFFER_BVH_PRIMITIVE_INDICES_LOOKUP_BUFFER].Load<uint>((node.first_index+i)*4);
                PrimitiveNode node = storage_buffers_readonly[STORAGE_BUFFER_PRIMITIVE_FLAT_LIST_BUFFER].Load<PrimitiveNode>(node_index * sizeof(PrimitiveNode));

                // max value is special case of a "ghost" primitive (that has a BVH node but does not actually do anything)
                if (node.type != uint.maxValue) {
                    apply_primitive_sdf(p, node, node_index, sdf);
                }
            }
        }
    }
    
    return sdf;
}

void capture_performance_bvh_distance_query_inner_loop() {
    if (is_bit_set(SPEC_CONSTANTS_BITFIELD, 0)) {
        storage_buffers[STORAGE_BUFFER_READBACK_BUFFER].InterlockedAdd(READBACK_BUFFER_SDF_BVH_INNER_LOOP_CALL_QUERY_INDEX * 4, 1);
    }
}

void capture_performance_sdf_primitive_call() {
    if (is_bit_set(SPEC_CONSTANTS_BITFIELD, 0)) {
        storage_buffers[STORAGE_BUFFER_READBACK_BUFFER].InterlockedAdd(READBACK_BUFFER_SDF_PRIMITIVE_CALLS_QUERY_INDEX * 4, 1);
    }
}

void capture_performance_query_trace_ray() {
    if (is_bit_set(SPEC_CONSTANTS_BITFIELD, 0)) {
        storage_buffers[STORAGE_BUFFER_READBACK_BUFFER].InterlockedAdd(READBACK_BUFFER_RAYS_TRACED_QUERY_INDEX * 4, 1);
    }
}

void capture_performance_sdf_call() {
    if (is_bit_set(SPEC_CONSTANTS_BITFIELD, 0)) {
        storage_buffers[STORAGE_BUFFER_READBACK_BUFFER].InterlockedAdd(READBACK_BUFFER_SDF_CALLS_QUERY_INDEX, 1);
    }
}

struct TraceDebugInfo {
    int iters;
    int3 chunk;
}

struct TraceRes {
    bool hit;
    float3 position;
    float3 normal;
    float closest_distance;
    TraceDebugInfo debug;
};

TraceRes trace(float3 ray_pos, float3 ray_dir) {
    float3 point = ray_pos;
    capture_performance_query_trace_ray();
    int total_iters = 0;
    float closest_distance_to_surface = 1;
    
    float3 modified_ray_pos = point / CHUNK_PHYSICAL_SIZE;
    int3 floored_pos = floor(modified_ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    int3 eqs;
    float3 side_dist = (floored_pos - modified_ray_pos + 0.5 + 0.5 * dir_sign);
    float closest_distance = 1.0;

    outer: for (int i = 0; i < DDA_TRAVERSAL_CHUNKS_ITER; i++) {

        uint chunk_start_index = storage_images_3d_uint[CHUNK_LOOKUP_TEXTURE].Load(floored_pos + (int)CHUNK_LOOKUP_MEGA_TEXTURE_HALF_SIZE);
        
        // make sure texel points to an actual chunk that has been loaded/generated
        if (chunk_start_index != uint.maxValue) {
            float3 test = (floored_pos - modified_ray_pos + 0.5 - 0.5 * dir_sign) * inv_dir;
            float d = max3(test.x, test.y, test.z);
            float3 point = (modified_ray_pos + ray_dir * max(d, 0)) * CHUNK_PHYSICAL_SIZE;
            // point += ray_dir*CHUNK_BOUNDARY_OFFSET_EPS;
            
            // FIXME: there is still an issue here, when the chunk get further away from the origin (and thus getting more fp precision issues)
            // temporarily fixed this by increasing CHUNK_BOUNDARY_OFFSET_EPS, but that isn't really a sollution
            point = clamp(point, floored_pos * CHUNK_PHYSICAL_SIZE + CHUNK_BOUNDARY_OFFSET_EPS, (floored_pos+1) * CHUNK_PHYSICAL_SIZE - CHUNK_BOUNDARY_OFFSET_EPS);
            
            inner: for (int o = 0; o < DDA_TRAVERSAL_RAY_MARCH_ITER; o++) {
                // check if we moved from one chunk to another chunk
                // if we did, then we must stop raymarching the current chunk
                if (any((floor(point / CHUNK_PHYSICAL_SIZE) != (float3)floored_pos))) {
                    break inner;
                }

                // SDF stored in x, normal stored in yzw
                float4 sampled = sample_chunk(point, chunk_start_index);
                closest_distance = min(closest_distance, sampled.x);

                // check if we got close enough to the surface of the SDF
                if (sampled.x < THRESHOLD_DISTANCE) {
                    return TraceRes(true, point, sampled.yzw, closest_distance, TraceDebugInfo(total_iters, floored_pos));
                }
            
                point += ray_dir * sampled.x;
                total_iters++;

            }
        } 
    
        float3 reconst = side_dist * inv_dir;
        eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    return TraceRes(false, 0, 0, closest_distance, TraceDebugInfo(total_iters, 0));
}


float shadow_trace(float3 ray_pos, float3 ray_dir) {
#ifdef SHADOWS
    return 1-trace(ray_pos, ray_dir).hit;
#else
    return 1.0;
#endif
}

#endif