#ifndef SDF
#define SDF

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"
#include "sdf_iq.slang"
#include "sdf_ext.slang"


struct Node {
    uint transform_index;
    uint type;
}

enum SdfType {
    Box,
    BoxWithCutout,
    Rock,
    Interesting1,
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
    }

    return 0.0;
}

struct Node3D {
    float16_t3 min;
    uint16_t prim_count;
    float16_t3 max;
    uint16_t first_index;
};

#define CACHE_MY_SHIT

float3 get_normal_raw(float3 p) {
    float eps = 0.001;
    float x = d(p - float3(eps, 0, 0)).x;
    float y = d(p - float3(0, eps, 0)).x;
    float z = d(p - float3(0, 0, eps)).x;
    float b = d(p).x;
    return normalize(float3(b-x, b-y, b-z));
}



// wtf the difference between using this vs the auto-diff normal is NUTS
// at 720p, this changes the frame timing (w path-tracing) from 12ms to 5ms. wtf
float3 get_normal_test(float3 p) {
    //  get_normal_raw(p);
    // return 0.0;
#ifdef CACHE_MY_SHIT
    return sample_poop2_normals(p);
    //return sampled_images_3d[VXGI_SAMPLED_IMAGE_IDX].SampleLevel(samplers[LINEAR_CLAMPED_SAMPLER_IDX], ((p.xyz * 4) / VXGI_TEXTURE_SIZE) + 0.5, 0).yzw;
#else
    return get_normal_raw(p);
#endif
}

static int sampled_from_raw_sdf = 0;
static int sampled_from_cached_sdf = 0;

float d2(float3 p) {
    
#ifdef CACHE_MY_SHIT
    int3 chunk_pos = (int3)floor(p / CHUNK_PHYSICAL_SIZE);

    if (any(chunk_pos < -30 | chunk_pos > 30)) {
        return 10000;
    }

    uint chunk_start_index = storage_images_3d_uint[CHUNK_LOOKUP_TEXTURE].Load(chunk_pos + CHUNK_LOOKUP_MEGA_TEXTURE_HALF_SIZE);
    

    if (chunk_start_index == uint.maxValue) {
        // return 0;
        return 0;
        sampled_from_raw_sdf++;
        // return d(p).x;
    } 

    uint target_texture_index = storage_buffers_readonly[STORAGE_BUFFER_CHUNKS_BUFFER].Load<uint>(chunk_start_index * 4);
    float3 local_position = p - floor(p / CHUNK_PHYSICAL_SIZE) * CHUNK_PHYSICAL_SIZE;
    float4 poop =  sample_non_uniform_3d(target_texture_index, samplers[LINEAR_CLAMPED_SAMPLER_IDX], (local_position) / CHUNK_PHYSICAL_SIZE);
    sampled_from_cached_sdf++;
    return poop.x;
    //return sampled_images_3d[VXGI_SAMPLED_IMAGE_IDX].SampleLevel(samplers[LINEAR_CLAMPED_SAMPLER_IDX], (((p.xyz ) * 4) / VXGI_TEXTURE_SIZE) + 0.5, 0).x;
#else
    // return sample_poopy2(p);
    return d(p).x;
#endif
}

float2 d(float3 p) {
    capture_performance_sdf_call();


    float sdf = p.y;
    float material = 0.0;

    //sdf = min(sdf, sample_poopy2(p));
        
    uint len = 0;
    const static uint capacity = 10;
    uint[capacity] arrays = { 0 };

    len += 1;
    
    while (len > 0) {
        capture_performance_bvh_distance_query_inner_loop();
        len -= 1;
        uint index = arrays[len];
        Node3D node = storage_buffers_readonly[STORAGE_BUFFER_WOW1_BUFFER].Load<Node3D>(sizeof(Node3D) * index);

        float3 aabb_min = node.min.xyz;
        float3 aabb_max = node.max.xyz;
        float3 center = (aabb_min+aabb_max)*0.5;
        float3 half_extent = (aabb_max-aabb_min) * 0.5;
        float bvh_node_sdf = sdBox(p-center, half_extent);

        if (node.prim_count == 0) {
            if (sdf > bvh_node_sdf) {
                if (len < capacity) {
                    Node3D first_child = storage_buffers_readonly[STORAGE_BUFFER_WOW1_BUFFER].Load<Node3D>(sizeof(Node3D) * node.first_index);

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
            if (sdf < bvh_node_sdf) {
                continue;
            }

            for (int i = 0; i < node.prim_count; i++) {
                capture_performance_sdf_primitive_call();
                uint node_index = storage_buffers_readonly[STORAGE_BUFFER_WOW2_BUFFER].Load<uint>((node.first_index+i)*4);
                uint node_byte_offset = node_index * sizeof(Node); 
                Node node = storage_buffers_readonly[STORAGE_BUFFER_PRIMITIVE_FLAT_LIST_BUFFER].Load<Node>(node_byte_offset);
                let tmp = sdf;

                if (node.type != uint.maxValue) {
                    // 3x4 = 3 rows, 4 columns
                    // matrix must be the INVERSE of the transform matrix
                    uint transform_byte_offset = node.transform_index * 4 * 4 * 3;
                    float3x4 transform = storage_buffers_readonly[STORAGE_BUFFER_TRANSFORMS_BUFFER].Load<float3x4>(transform_byte_offset);

                    // 3x4 * 4x1 = 3x1 vector
                    float3 local = mul(transform, float4(p, 1));

                    SdfType sdf_type = SdfType(node.type);
                    // SdfType sdf_type = SdfType.Rock;
                    
                    float obj_sdf = get_sdf_from_type(local.xyz, sdf_type, node_index); 
                    // sdf = smin(sdf, obj_sdf, 0.1);
                    // sdf = max(sdf, -obj_sdf);

                    sdf = min(sdf, obj_sdf);

                    if (sdf < 0) {
                        // return sdf;
                    }

                    if (obj_sdf < tmp) {
                        material = 1.0;
                    }
                }
            }
        }
    }

    /*
    // contains up to 4 intersecting nodes, packed as a byte each.
    // packed as linked list, from least-significant byte
    uint3 offset = (uint3)floor(p + 64);
    uint packed4_node_indices = storage_images_3d_uint[LOOKUP_TEXTURE].Load(offset);

    if (packed4_node_indices != 0) {

        // compiler seems to shit itself when you try to exit early from this loop
        // unsure if its the driver compiler or slang that causes this problem
        [ForceUnroll]
        for (int k = 0; k < 3; k++) {
            uint node_index = (packed4_node_indices >> 8 * k) & 0xFF;

            if (node_index != 0) {
                uint node_byte_offset = node_index * sizeof(Node); 
                Node node = storage_buffers[STORAGE_BUFFER_PRIMITIVE_FLAT_LIST_BUFFER].Load<Node>(node_byte_offset);
                let tmp = sdf;


                // 3x4 = 3 rows, 4 columns
                // matrix must be the INVERSE of the transform matrix
                uint transform_byte_offset = node.transform_index * 4 * 4 * 3;
                float3x4 transform = storage_buffers[STORAGE_BUFFER_TRANSFORMS_BUFFER].Load<float3x4>(transform_byte_offset);

                // 3x4 * 4x1 = 3x1 vector
                float3 local = mul(transform, float4(p, 1));

                float obj_sdf = get_sdf_from_type(local.xyz, SdfType.Rock, node_index); 
                // sdf = min(sdf, obj_sdf);
                sdf = smin(sdf, obj_sdf, 0.1);
                // sdf = max(sdf, -obj_sdf);

                if (obj_sdf < tmp) {
                    material = 1.0;
                }
            } else {
                break;
            }
        }
    }
    */

    // sdf = sdFbm(p / 1000.0, p.y / 1000.0) * 1000.0;
    // sdf = smin(sdf, sdOctahedron(p, 1), 0.1);
    // sdf = smin(sdf, sdOctahedron(p-float3(0, 1, 0), 1), 0.1);
    // sdf = smin(sdf, sdOctahedron(p-float3(0, 2, 0), 1), 0.1);
    
    
    
    return float2(sdf, material);
}

struct TraceResHitObj {
    int instance_custom_index;
    float material;
}

struct TraceRes {
    int iters;
    bool hit;
    float3 pos;
    float3 dir;
    float3 normal;
    int sdf_calls;
    TraceResHitObj data;
};

struct ShadowTraceRes {
    float shadow;
};

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

#define HWRT_BOUNDS_ACCEL
#define HWRT_BOUNDS_ACCEL_SHADOWS
#define SHADOWS

static const uint MAX_ITERS = 256;
static const float MAX_DISTANCE = 10000;
static const float THRESHOLD_DISTANCE_PREPASS = 0.01;
static const float THRESHOLD_DISTANCE = 0.001;
static const float DISTANCE_SCALING_FACTOR = 0.9;


float2 ray_box_get_tmin_tmax(
    float3 ray_pos,
    float3 ray_dir,
    float3 aabb_min,
    float3 aabb_max,
) {
    // TODO: optimize if required
    float3 t1 = (aabb_min - ray_pos) / ray_dir;
    float3 t2 = (aabb_max - ray_pos) / ray_dir;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);
    return float2(tmin,tmax);
}

const static uint INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK = 1 << 20;
const static uint INSTANCE_CUSTOM_INDEX_AABB_START_INDEX = (1 << 5) - 1;
const static uint INSTANCE_CUSTOM_INDEX_CHUNK_SDF_FLAG_MASK = 1 << 5;
const static uint INSTANCE_CUSTOM_INDEX_CHUNK_SDF_MASK = ((1 << 20) - 1) & ~((1 << 5) - 1);


struct GpuAabb {
    half4 min;
    half4 max;
};

const static float CHUNK_PHYSICAL_SIZE = 4.0;
const static float CHUNK_LOGICAL_SIZE = 64;

const static uint CHUNK_LOOKUP_MEGA_TEXTURE_SIZE = 128;
const static uint CHUNK_LOOKUP_MEGA_TEXTURE_HALF_SIZE = CHUNK_LOOKUP_MEGA_TEXTURE_SIZE / 2;

float sample_poopy2(float3 p) {

    int3 chunk_pos = (int3)floor(p / CHUNK_PHYSICAL_SIZE);
    uint chunk_start_index = storage_images_3d_uint[CHUNK_LOOKUP_TEXTURE].Load(chunk_pos + CHUNK_LOOKUP_MEGA_TEXTURE_HALF_SIZE);
    

    if (chunk_start_index == uint.maxValue) {
        return 100000;
    } 

    uint target_texture_index = storage_buffers_readonly[STORAGE_BUFFER_CHUNKS_BUFFER].Load<uint>(chunk_start_index * 4);
    float3 local_position = p - floor(p / CHUNK_PHYSICAL_SIZE) * CHUNK_PHYSICAL_SIZE;
    float4 poop =  sample_non_uniform_3d(target_texture_index, samplers[LINEAR_CLAMPED_SAMPLER_IDX], (local_position) / CHUNK_PHYSICAL_SIZE);
    return poop.x;
}

float3 sample_poop2_normals(float3 p)
{
    int3 chunk_pos = (int3)floor(p / CHUNK_PHYSICAL_SIZE);
    uint chunk_start_index = storage_images_3d_uint[CHUNK_LOOKUP_TEXTURE].Load(chunk_pos + CHUNK_LOOKUP_MEGA_TEXTURE_HALF_SIZE);
    

    if (chunk_start_index == uint.maxValue) {
        return 100000;
    } 

    uint target_texture_index = storage_buffers_readonly[STORAGE_BUFFER_CHUNKS_BUFFER].Load<uint>(chunk_start_index * 4);
    float3 local_position = p - floor(p / CHUNK_PHYSICAL_SIZE) * CHUNK_PHYSICAL_SIZE;
    float4 poop =  sample_non_uniform_3d(target_texture_index, samplers[LINEAR_CLAMPED_SAMPLER_IDX], (local_position) / CHUNK_PHYSICAL_SIZE);
    return poop.yzw;
    /*
    float eps = 0.01;
    float x = sample_poopy2(p - float3(eps, 0, 0)).x;
    float y = sample_poopy2(p - float3(0, eps, 0)).x;
    float z = sample_poopy2(p - float3(0, 0, eps)).x;
    float b = sample_poopy2(p).x;


    return normalize(float3(b-x, b-y, b-z) / eps);
    */
}

[BackwardDerivativeOf(sample_poopy2)]
void bwd_sample_poopy2(inout DifferentialPair<float3> p_pair, float dOut)
{
    float eps = 0.01;
    float x = sample_poopy2(p_pair.p - float3(eps, 0, 0)).x;
    float y = sample_poopy2(p_pair.p - float3(0, eps, 0)).x;
    float z = sample_poopy2(p_pair.p - float3(0, 0, eps)).x;
    float b = sample_poopy2(p_pair.p).x;


    p_pair = diffPair<float3>(p_pair.p, (float3(b-x, b-y, b-z) / eps) * dOut.x);
}

void trace_inner<uint T>(inout RayQuery<T> sq, inout int i, inout int sdf_calls, inout float material) {
    // 24 bits
    uint instance_custom_index = sq.CandidateRayInstanceCustomIndex();   
    bool local_sdf = (instance_custom_index & INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK) != 0;
    uint aabb_lookup_index = (instance_custom_index & INSTANCE_CUSTOM_INDEX_AABB_START_INDEX) + sq.CandidatePrimitiveIndex();

    // software AABB test lmao to get min and max dists of ray into  AABB
    GpuAabb aabb = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<GpuAabb>(aabb_lookup_index * sizeof(GpuAabb));
    float2 tmin_tmax = ray_box_get_tmin_tmax(sq.CandidateRayObjectRayOrigin(), sq.CandidateObjectRayDirection(), aabb.min.xyz, aabb.max.xyz);
    float distance = max(tmin_tmax.x, 0);

    // cache ray data
    float3 ro = sq.WorldRayOrigin();
    float3 rd = sq.WorldRayDirection();
    float3 point = ro + rd * distance;

    tmin_tmax.y = min(tmin_tmax.y, sq.CommittedRayT());

    for (; i < MAX_ITERS; i++) {
        // shaves off a few ms
        if (distance > tmin_tmax.y) {
            return;
        }

        capture_performance_sdf_call();
    
        float2 closest_distance_and_material = 0;
        if (!local_sdf) {
           closest_distance_and_material = d(point);
        } else {
            // TODO: it would be smarter to convert the point to object space first and trace using that
            // atp just duplicate trace_inner to a "local" version that takes in a local SDF instead of global one
            float3 local_position = mul(sq.CandidateWorldToObject3x4(), float4(point, 1)).xyz;
            float instance_index = sq.CandidateInstanceIndex();
            // closest_distance_and_material = tree2(local_position * 2.5, instance_index) / 2.5;
            closest_distance_and_material = sample_tree(local_position);
            //closest_distance_and_material = float2(length(local_position) - 0.5, 1.0);
        }
        sdf_calls += 1;

        // closest_distance_and_material.x = max(closest_distance_and_material.x, -(length(point - (uniforms.position.xyz + uniforms.forward.xyz * 10)) - 2));
        // check if we got close enough to the surface
        if (closest_distance_and_material.x < THRESHOLD_DISTANCE) {
            sq.CommitProceduralPrimitiveHit(distance); 
            material = closest_distance_and_material.y;
            return;
        }

        float step_size = closest_distance_and_material.x * DISTANCE_SCALING_FACTOR;
        point += rd * step_size;
        distance += step_size;
    }
}



float trace_get_closest_to_surface(float3 ray_pos, float3 ray_dir) {
    capture_performance_query_trace_ray();

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> sq;
    let flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

    RayDesc desc;		

    desc.Origin = ray_pos;
    desc.Direction = ray_dir;
    desc.TMax = MAX_DISTANCE;
    desc.TMin = 0.0;

    sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);

    float d = 100000;

    while (sq.Proceed()) {
        if(sq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {

            // 24 bits
            uint instance_custom_index = sq.CandidateRayInstanceCustomIndex();   
            bool local_sdf = (instance_custom_index & INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK) != 0;
            uint aabb_lookup_index = (instance_custom_index & INSTANCE_CUSTOM_INDEX_AABB_START_INDEX) + sq.CandidatePrimitiveIndex();

            // software AABB test lmao to get min and max dists of ray into  AABB
            GpuAabb aabb = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<GpuAabb>(aabb_lookup_index * sizeof(GpuAabb));
            float2 tmin_tmax = ray_box_get_tmin_tmax(sq.CandidateRayObjectRayOrigin(), sq.CandidateObjectRayDirection(), aabb.min.xyz, aabb.max.xyz);
            float distance = max(tmin_tmax.x, 0);

            if (distance < d) {
                sq.CommitProceduralPrimitiveHit(distance); 
                d = distance;
            }
        }
    }

    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        return sq.CommittedRayT();
    } else {
        return 100000;
    }
}



TraceRes trace(float3 ray_pos, float3 ray_dir) {
    capture_performance_query_trace_ray();

#ifdef HWRT_BOUNDS_ACCEL

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> sq;
    let flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

    RayDesc desc;		

    desc.Origin = ray_pos;
    desc.Direction = ray_dir;
    desc.TMax = MAX_DISTANCE;
    desc.TMin = 0.0;

    sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);
    int i = 0;
    int sdf_calls = 0;
    float material = 0;

    while (sq.Proceed()) {
        if(sq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {

            trace_inner(sq, i, sdf_calls, material);
            
            if (i == MAX_ITERS) {
                sq.Abort();
            }
        }
    }

    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        float3 point = sq.WorldRayOrigin() + sq.WorldRayDirection() * sq.CommittedRayT();
        float3 normal = 0.0;

        // 24 bits
        uint instance_custom_index = sq.CommittedRayInstanceCustomIndex();   
        bool local_sdf = (instance_custom_index & INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK) != 0;
        
        if (!local_sdf) {
            DifferentialPair<float3> dp_a = diffPair(point);
            //bwd_diff(d)(dp_a, 1.0);
            // normal = normalize(dp_a.d); 
            normal = get_normal_test(point);
        } else {
            DifferentialPair<float3> dp_a = diffPair(mul(sq.CommittedWorldToObject3x4(), float4(point, 1)).xyz);
            DifferentialPair<float> dp_i = diffPair(sq.CommittedInstanceIndex());
            bwd_diff(sample_tree)(dp_a, 1.0);  
            
            float3 object_normal = normalize(dp_a.d);
            normal = mul(sq.CommittedObjectToWorld3x4(), float4(object_normal, 0)).xyz;
        }

        return TraceRes(i, true, point, ray_dir, normal, sdf_calls, TraceResHitObj((0), material));
    } else {
        return TraceRes(i, false, ray_pos, ray_dir, 0.0, sdf_calls, TraceResHitObj());
    }
#else
    float3 point = ray_pos;
    int i = 0;

    // experiment 2026-08-29: implemented outer DDA loop to see if that would speed up anything
    // it does seem to do *something*, but still too slow relative to HWRT traversal. this helped pin-point a performance bottleneck: calculating the normal
    // auto-diff normal is really slow.

    // neat trick: threshold increase proportionally to distance AND pixel "factor" which takes partial derivative of ray dir (to give a sense of how "large" a ray is)
    // float pixel_thing = max(length(ddx(ray_dir)), length(ddy(ray_dir)));
    for (; i < MAX_ITERS; i++) {
        
        capture_performance_sdf_call();
    

        let k = d(point).x;

        if (k < THRESHOLD_DISTANCE) {
            let normal = get_normal_test(point);
            return TraceRes(i, true, point, ray_dir, normal, i);
        }

        point += ray_dir * k;
    }

    return TraceRes(i, false, ray_pos, ray_dir, 0.0, i);
#endif
}


ShadowTraceRes shadow_trace(float3 ray_pos, float3 ray_dir) {
#ifndef SHADOWS
    return ShadowTraceRes(1.0);
#endif

    capture_performance_query_trace_ray();

#ifdef HWRT_BOUNDS_ACCEL_SHADOWS
    // 7.7ms vs 7.3ms when using RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH at 800x600, 1 bounce, default scene
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    let flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    RayDesc desc;		

    desc.Origin = ray_pos;
    desc.Direction = ray_dir;
    desc.TMax = MAX_DISTANCE;
    desc.TMin = 0.0;

    sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);

    int i = 0;
    int sdf_calls = 0;
    float material = 0;
    while (sq.Proceed()) {
        if(sq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            trace_inner(sq, i, sdf_calls, material);
            if (i == MAX_ITERS) {
                sq.Abort();
            }
        }
    }

    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        return ShadowTraceRes(0.0);
    } else {
        return ShadowTraceRes(1.0);
    }
#else
    #define SDF_SOFT_SHADOWS_SUPPORTED
    float shadow = 1.0;
    float distance = 0.05;

    for (int i = 0; i < MAX_ITERS; i++) {
        capture_performance_sdf_call();
    

        let k = d(ray_pos + ray_dir * distance).x;

        if (k < THRESHOLD_DISTANCE) {
            return ShadowTraceRes(0.0);
        }

        // I have no idea why using `distance` here instead of `k` fixes the issue with the "overcast" issue (where there seemed to be penumbra everywhere)
        // if I had to guess, distance reaches MAX_DISTANCE much more quicker than k reaches MAX_DISTANCE, and breaks from the loop earlier
        // if the loop were to continue, at some point, the map itself would be "casting" a shadow upon itself because it would consider it as part of a very big penumbra (even though the map is *behind* the shadow ray,
        // something I thought would be fixable by taking gradient each step and comparing it with the sun direction)
        
        // this is a much easier fix though
        if (distance > MAX_DISTANCE) {
            break;
        }

        // https://iquilezles.org/articles/rmshadows/
        shadow = min(shadow, saturate(k * 8.0 / distance));
        distance += k;
    }

    return ShadowTraceRes(shadow);
#endif
}

#endif