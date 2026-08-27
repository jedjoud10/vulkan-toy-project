#ifndef SDF
#define SDF

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"
#include "sdf_iq.slang"
#include "sdf_ext.slang"

[Differentiable]
float2 d(float3 p) {
    float t = uniforms.time;

    // float s = length(p - 10*clamp(round(p/10.0), -float3(100, 1, 100), float3(100, 1, 100))) - 2;
    float sphere = length(p + float3(0, -1, 0)) - 2;
    float octa = sdOctahedron(p - float3(0, 20, 0), 2.0);
    
    //float a = sdHexPrism(p + float3(-10, -2, 0,), float2(2, 4));
    //float b = sdHexPrism(p + float3(-10, -2, 0,), float2(1, 5));

    //float hex = max(-b, a);
    float hex = sdBox(p + float3(-10, -2, 0,), float3(2,2,4));
    float cyl = sdCylinder(p + float3(10, 0, 0), float3(0, 0, 2));

    float3x3 rot = AngleAxis3x3(2.4, normalize(float3(1, 1, 1)));
    float3 torus_origin = float3(2, 2, 5);
    float tor = sdTorus(mul(p - torus_origin, rot), float2(5, 2)) + sin(p.y * 20.0) * 0.05 + 0.05;

    float ground = p.y /* + snoise(p.xz * 2.0) * 0.2 + 0.1 */;
    //ground = max(ground, tex_sample(p * 50.0 - 4.2312) / 50.0);
    //ground = min(ground, p.y+0.8);

    float sdf = min(sphere, min(ground, min(min(octa, cyl), min(hex, tor))));

    /*
    for (int i = 0; i < 0; i++) {
        Primitive prim = storage_buffers_readonly[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS].Load<Primitive>(i * 16);
    
        //float primitive_sdf = length(p - prim.position) - 1.0;
        float primitive_sdf = sdTorus(p - prim.position, float2(0.5, 0.25));

        if (prim.flags == 1) {
            sdf = min(sdf, primitive_sdf);
        } else {
            sdf = max(sdf, -primitive_sdf);
        }
    }
    */

    float other = tex_sample(p);
    
    float mat = 0.0;
    
    if (other < sdf) {
        mat = 1.0;    
    }

    sdf = smin(sdf, other, SMOOTHING_FACTOR_MAP_TEXTURE);

    //sdf = smin(sdf, d2(p), 0.5);
    // sdf = min(sdf, tex_sample(p - float3(0, 10, 0)));
    return float2(sdf, mat);
}

struct Primitive {
    float3 position;
    uint flags;
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
static const float THRESHOLD_DISTANCE = 0.001;
static const float DISTANCE_SCALING_FACTOR = 0.9;
static const bool LOD = true;


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

const static uint INSTANCE_CUSTOM_INDEX_AABB_LOOKUP_INDEX_MASK = 15;
const static uint INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK = 16;
const static uint INSTANCE_CUSTOM_INDEX_LOOKUP_AABB_USING_PRIMITIVE_INDEX_FLAG_MASK = 32;


void trace_inner<uint T>(inout RayQuery<T> sq, inout int i, inout int sdf_calls, inout float material) {
    /*
    if (!WaveActiveAllEqual(sq.CandidateInstanceIndex())) {
        if (WaveActiveCountBits(true) < 50) {
            return;
        }
    }
    */

    // 24 bits
    uint instance_custom_index = sq.CandidateRayInstanceCustomIndex();

    // extract flags
    bool local_sdf = (instance_custom_index & INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK) != 0;
    bool aabb_index_from_primitive_index = (instance_custom_index & INSTANCE_CUSTOM_INDEX_LOOKUP_AABB_USING_PRIMITIVE_INDEX_FLAG_MASK) != 0;
    uint aabb_lookup_index = select(aabb_index_from_primitive_index, sq.CandidateRayPrimitiveIndex(), instance_custom_index & INSTANCE_CUSTOM_INDEX_AABB_LOOKUP_INDEX_MASK);

    float local_distance_scaling_factor = select(local_sdf, 0.8, 1.0);

    // software AABB test lmao to get min and max dists of ray into  AABB
    float3 min_bounds = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<float3>(aabb_lookup_index * 3*4*2);
    float3 max_bounds = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<float3>(aabb_lookup_index * 3*4*2 + 3*4);
    float2 tmin_tmax = ray_box_get_tmin_tmax(sq.CandidateRayObjectRayOrigin(), sq.CandidateObjectRayDirection(), min_bounds, max_bounds);
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
        }
        sdf_calls += 1;

        // closest_distance_and_material.x = max(closest_distance_and_material.x, -(length(point - (uniforms.position.xyz + uniforms.forward.xyz * 10)) - 2));
        // check if we got close enough to the surface
        if (closest_distance_and_material.x < select(LOD, max(THRESHOLD_DISTANCE * distance, THRESHOLD_DISTANCE), THRESHOLD_DISTANCE)) {
            sq.CommitProceduralPrimitiveHit(distance); 
            material = closest_distance_and_material.y;
            return;
        }

        float step_size = closest_distance_and_material.x * DISTANCE_SCALING_FACTOR * local_distance_scaling_factor;
        point += rd * step_size;
        distance += step_size;
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
        if ((sq.CommittedRayInstanceCustomIndex() & INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK) == 0) {
            DifferentialPair<float3> dp_a = diffPair(point);
            bwd_diff(d)(dp_a, 1.0);
            normal = normalize(dp_a.d); 
        } else {
            DifferentialPair<float3> dp_a = diffPair(mul(sq.CommittedWorldToObject3x4(), float4(point, 1)).xyz);
            DifferentialPair<float> dp_i = diffPair(sq.CommittedInstanceIndex());
            bwd_diff(sample_tree)(dp_a, 1.0);  
            
            float3 object_normal = normalize(dp_a.d);
            normal = mul(sq.CommittedObjectToWorld3x4(), float4(object_normal, 0)).xyz;
        }

        return TraceRes(i, true, point, ray_dir, normal, sdf_calls, TraceResHitObj((sq.CommittedRayInstanceCustomIndex() & INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK), material));
    } else {
        return TraceRes(i, false, ray_pos, ray_dir, 0.0, sdf_calls, TraceResHitObj());
    }
#else
    float3 point = ray_pos;
    int i = 0;

    // neat trick: threshold increase proportionally to distance AND pixel "factor" which takes partial derivative of ray dir (to give a sense of how "large" a ray is)
    // float pixel_thing = max(length(ddx(ray_dir)), length(ddy(ray_dir)));
    for (; i < MAX_ITERS; i++) {
        
        capture_performance_sdf_call();
    

        let k = d(point);

        if (k < THRESHOLD_DISTANCE) {
            // if (k < max(THRESHOLD_DISTANCE * distance * pixel_thing * 1000.0, THRESHOLD_DISTANCE)) {
            DifferentialPair<float3> dp_a = diffPair(point);
            
            bwd_diff(d)(dp_a, 1.0);  
            float3 normal = normalize(dp_a.d);
            return TraceRes(i, true, point, ray_dir, normal, i);
        }

        if (k > MAX_DISTANCE) {
            break;
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
    

        let k = d(ray_pos + ray_dir * distance);

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