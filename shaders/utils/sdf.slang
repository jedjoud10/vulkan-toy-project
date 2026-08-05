#ifndef SDF
#define SDF

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"

// https://iquilezles.org/articles/smin/
[Differentiable]
// cubic polynomial
float smin( float a, float b, float k )
{
    k *= 6.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*h*k*(1.0/6.0);
}

// https://iquilezles.org/articles/distfunctions/
[Differentiable]
float sdOctahedron( vec3 p, float s )
{
  p = abs(p);
  float m = p.x+p.y+p.z-s;
  vec3 q;
       if( 3.0*p.x < m ) q = p.xyz;
  else if( 3.0*p.y < m ) q = p.yzx;
  else if( 3.0*p.z < m ) q = p.zxy;
  else return m*0.57735027;
    
  float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
  return length(vec3(q.x,q.y-s+k,q.z-k)); 
}

// https://iquilezles.org/articles/distfunctions/
[Differentiable]
float sdHexPrism( vec3 p, vec2 h )
{
  const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
  p = abs(p);
  p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
  vec2 d = vec2(
       length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// https://iquilezles.org/articles/distfunctions/
[Differentiable]
float sdCylinder( vec3 p, vec3 c )
{
  return length(p.xz-c.xy)-c.z;
}

// https://iquilezles.org/articles/distfunctions/
[Differentiable]
float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

// https://iquilezles.org/articles/distfunctions/
[Differentiable]
float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b + r;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

// https://iquilezles.org/articles/distfunctions/
[Differentiable]
float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}



const static int TEXTURE_SIZE = 128;
float tex_sample(float3 p) {
    return sampled_images_3d[SDF_TEXTURE_IMAGE_IDX].Sample(samplers[LINEAR_UNCAMPLED_SAMPLER_IDX], p / TEXTURE_SIZE + 0.5).x;
}

[BackwardDerivativeOf(tex_sample)]
void bwd_tex_sample(inout DifferentialPair<float3> p_pair, float dOut)
{
    float eps = 0.1;
    float x = tex_sample(p_pair.p - float3(eps, 0, 0));
    float y = tex_sample(p_pair.p - float3(0, eps, 0));
    float z = tex_sample(p_pair.p - float3(0, 0, eps));
    float b = tex_sample(p_pair.p);


    p_pair = diffPair<float3>(p_pair.p, (float3(b-x, b-y, b-z) / eps) * dOut);
}

[Differentiable]
float d(float3 p) {
    float t = uniforms.time;

    // float s = length(p - 10*clamp(round(p/10.0), -float3(100, 1, 100), float3(100, 1, 100))) - 2;
    float sphere = length(p + float3(0, -1, 0)) - 2;
    float octa = sdOctahedron(p - float3(0, 20, 0), 2.0);
    
    float a = sdHexPrism(p + float3(-10, -2, 0,), float2(2, 4));
    float b = sdHexPrism(p + float3(-10, -2, 0,), float2(1, 5));

    float hex = max(-b, a);
    float cyl = sdCylinder(p + float3(10, 0, 0), float3(0, 0, 2));

    float3x3 rot = AngleAxis3x3(2.4, normalize(float3(1, 1, 1)));
    float3 torus_origin = float3(2, 2, 5);
    float tor = sdTorus(mul(p - torus_origin, rot), float2(5, 2)) + sin(p.y * 20.0) * 0.05 + 0.05;

    float ground = p.y /* + snoise(p.xz * 2.0) * 0.2 + 0.1 */;
    //ground = max(ground, tex_sample(p * 50.0 - 4.2312) / 50.0);
    //ground = min(ground, p.y+0.8);

    float sdf = min(sphere, min(ground, min(min(octa, cyl), min(hex, tor))));


    //sdf = smin(sdf, d2(p), 0.5);
    // sdf = min(sdf, tex_sample(p - float3(0, 10, 0)));
    return sdf;
}

[Differentiable]
float tree1(float3 lp) {
    float3 rotated = lp.xzy;
    
    // base trunk
    float d = sdHexPrism(rotated - float3(0, 0, -3.0), float2(0.5, 5));
    
    d = min(d, sdHexPrism(rotated - float3(0, 0, 0.5) + float3(sin(uniforms.time - 3.213), cos(uniforms.time + 4.534), 0) * 0.1, float2(2.0, 0.5)));
    d = min(d, sdHexPrism(rotated - float3(0, 0, 4.5) + float3(sin(uniforms.time), cos(uniforms.time), 0) * 0.1, float2(1.5, 3.5)));
    
    return d;
}

[Differentiable]
float2 tree2(float3 lp, float instance_index) {
    
    // half extent
    float box_height = 4;
    
    // branch space
    float3 bp = lp - float3(0, -2.0 * box_height, 0);
    float d = 10000;
    float thickness = 0.7;
    
    //float hash_like = sin(i * 432.34 + 4.32) * 4.324;
    float3x3 local_rot = AngleAxis3x3(0.2 + sin(instance_index) * 0.03, float3(1, 0, -1));

    for (int i = 0; i < 4; i++) {
        bp.xz = abs(bp.xz);

        // box is center at (0, 1, 0) in local branch space
        // the origin (bp = (0, 0, 0)) is the BOTTOM of the box
        d = min(d, sdBox(bp - float3(0, box_height, 0), float3(thickness, box_height, thickness)));

        // move UP to reach end of box
        bp -= float3(0, 2 * box_height, 0);

        // rotate space
        bp = mul(bp, local_rot);
        
        // make next branches smaller and shorter
        thickness *= 0.8  + cos(34.2 * instance_index) * 0.1;
        box_height *= 0.6 + cos(3.2  * instance_index) * 0.1;
    }

    float id = 0;

    float sphere = length(bp) - 2.0;
    if (sphere < d) {
        d = sphere;
        id = 1;
    }
    
    //d = smin(d, , 0.2);

    return float2(d, id);
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

static const uint MAX_ITERS = 256;
static const float MAX_DISTANCE = 10000;
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

void trace_inner<uint T>(inout RayQuery<T> sq, inout int i, inout int sdf_calls) {    
    float3 aabb_symmetric_bounds = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<float3>(sq.CandidateRayInstanceCustomIndex() * 12);

    // software AABB test lmao to get min and max dists of ray into  AABB
    float2 tmin_tmax = ray_box_get_tmin_tmax(sq.CandidateRayObjectRayOrigin(), sq.CandidateObjectRayDirection(), -aabb_symmetric_bounds, aabb_symmetric_bounds);
    float distance = max(tmin_tmax.x, 0);
    
    float3 point = sq.WorldRayOrigin() + sq.WorldRayDirection() * distance;
    for (; i < MAX_ITERS; i++) {

        capture_performance_sdf_call();
    
        float k = 0;
        if (sq.CandidateRayInstanceCustomIndex() == 0) {
            k = d(point);
        } else {
            // TODO: it would be smarter to convert the point to object space first and trace using that
            // atp just duplicate trace_inner to a "local" version that takes in a local SDF instead of global one
            float3 local_position = mul(sq.CandidateWorldToObject3x4(), float4(point, 1)).xyz;
            float instance_index = sq.CandidateInstanceIndex();
            k = tree2(local_position, instance_index).x;
        }
        sdf_calls += 1;

        // k = max(k, -(length(point - (uniforms.position.xyz + uniforms.forward.xyz * 10)) - 2));
                        
        // check if we got close enough to the surface
        if (k < THRESHOLD_DISTANCE) {
            sq.CommitProceduralPrimitiveHit(distance); 
            return;
        }

        point += sq.WorldRayDirection() * k * DISTANCE_SCALING_FACTOR;
        distance += k * DISTANCE_SCALING_FACTOR;
    
        // check if we left the bounds of the AABB (tmax is calculated using software ray-AABB test)
        if (distance > tmin_tmax.y) {
            return;
        }

        // if we already have a closer intersection than this one, discard this one
        if (distance > sq.CommittedRayT()) {
            return;
        }
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
    while (sq.Proceed()) {
        if(sq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            
            trace_inner(sq, i, sdf_calls);
            
            if (i == MAX_ITERS) {
                sq.Abort();
            }
        }
    }

    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        float3 point = sq.WorldRayOrigin() + sq.WorldRayDirection() * sq.CommittedRayT();

        float material = 0;
        
        float3 normal = 0.0;
        if (sq.CommittedRayInstanceCustomIndex() == 0) {
            DifferentialPair<float3> dp_a = diffPair(point);
            bwd_diff(d)(dp_a, 1.0);
            normal = normalize(dp_a.d); 
        } else {
            DifferentialPair<float3> dp_a = diffPair(mul(sq.CommittedWorldToObject3x4(), float4(point, 1)).xyz);
            DifferentialPair<float> dp_i = diffPair(sq.CommittedInstanceIndex());
            bwd_diff(tree2)(dp_a, dp_i, 1.0);  

            material = tree2(dp_a.p, dp_i.p).y;
            
            float3 object_normal = normalize(dp_a.d);
            normal = mul(sq.CommittedObjectToWorld3x4(), float4(object_normal, 0)).xyz;
        }

        return TraceRes(i, true, point, ray_dir, normal, sdf_calls, TraceResHitObj(sq.CommittedRayInstanceCustomIndex(), material));
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
    capture_performance_query_trace_ray();

#ifdef HWRT_BOUNDS_ACCEL
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
    while (sq.Proceed()) {
        if(sq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            trace_inner(sq, i, sdf_calls);
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