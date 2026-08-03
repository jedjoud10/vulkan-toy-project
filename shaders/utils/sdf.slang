#ifndef SDF
#define SDF

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"

// https://iquilezles.org/articles/distfunctions/
[Differentiable]
float smin(float a, float b, float k) {
    float h = max(0., k-abs(b-a))/k;
    return min(a,b)-h*h*h*k/6.;
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

    float sdf = min(sphere, min(ground, min(min(octa, cyl), min(hex, tor))));

    float k = 1000;

    // TODO: fix this lolz
    uint count = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<uint32_t>(0);
    [MaxIters(3)]
    for (int element = 0; element < 3; element++) {
        float3 origin = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<float3>(4 + element * 12);
        k = smin(k, sdRoundBox(origin-p, 3.0, 1.0), 2);
    }

    sdf = max(sdf, -k);
    return sdf;
}

struct TraceRes {
    int iters;
    bool hit;
    float3 pos;
    float3 dir;
    float3 normal;
    float shadow;
    int sdf_calls;
};

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
    // software AABB test lmao to get min and max dists of ray into  AABB
    float2 tmin_tmax = ray_box_get_tmin_tmax(sq.CandidateRayObjectRayOrigin(), sq.CandidateObjectRayDirection(), -1.0, 1.0);
    float distance = max(tmin_tmax.x, 0);
    
    // tried implement a wave-based optimization, but due to the highly divergent rays this actually causes performance degradation (since there's overhead for nothing)
    // even with the primary rays (which are highly convergent) this still isn't helping 
    /*
    if (is_bit_set(uniforms.toggles_bitmask, 3)) {
        if (WaveActiveAllEqual(sq.CandidateInstanceIndex())) {
            float closest_ever = 2;
            float max_dist = 0;
            for (int quick=0; quick < 4; quick++) {
                let dquick = lerp(max(tmin_tmax.x, 0), max(tmin_tmax.y, 0), (float)quick / 3.0);
                float3 pquick = WaveActiveSum(sq.WorldRayOrigin() + sq.WorldRayDirection() * dquick) / (float)WaveActiveCountBits(true);
                float ev = d(pquick);
                sdf_calls += 1;
            
                if (ev < closest_ever) {
                    closest_ever = ev;

                    if (closest_ever > 0.0) {
                        distance = max(max(dquick, 0), tmin_tmax.x);
                    } else {
                        break;
                    }
                }
            }

            if (closest_ever > 0.4) {
                return;
            }
        }
    }
    */

    float3 point = sq.WorldRayOrigin() + sq.WorldRayDirection() * distance;
    for (; i < MAX_ITERS; i++) {
        float k = d(point);
        sdf_calls += 1;
        
        
        /*
        // 2nd test optimization
        float k = 0;
        if (is_bit_set(uniforms.toggles_bitmask, 3)) {
            if (!read_lds_cache(point, k)) {
                k = d(point);
                sdf_calls += 1;

                if (k > 0.5) {
                    store_lds_cache(point, k);
                }
            }
        } else {
            k = d(point);
            sdf_calls += 1;
        }
        */
        
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



// 32x32
static uint2 thread_id;

// 4x4
static uint2 wave_id_in_group;

// tried implementing some sort of SDF caching strategy so that coherent rays that access the SDF at similar points have a quick access cache instead of having to evaluate the SDF again
// works on the basis that the SDF can also be used as a lower bound to accelerating traversal 
// I have realized something
// this doesn't really help us because adjacent rays will STALL to wait to read the LDS cache (through read_lds_cache). They stall while they wait for a thread to compute the SDF for them. since wave size = group size in our case,
// this doesn't help for shit since the wave already executes in lockstep 
const static int cachesize = 32;

// uint stores x,y,z positions (int8, and floored distance as uint8)
groupshared uint lds_data[cachesize];

groupshared float lds_data2[cachesize];

int8_t3 discretize_position (float3 p) {
    return int8_t3(round(p));
}

uint index_hash(int8_t3 p) {
    return ((int)p.x * 6767 + (int)p.y * 51 + (int)p.z * 17) & (cachesize-1);
}

bool read_lds_cache(float3 position, out float distance) {
    GroupMemoryBarrierWithGroupSync();

    int8_t3 p = discretize_position(position);

    uint data = lds_data[index_hash(p)];
    
    int8_t4 loaded_p = (int8_t4)unpackInt4x8ToInt16(data);

    if (all(p.xyz == loaded_p.xyz)) {
        distance = lds_data2[index_hash(p)];
        //distance = 1000;
        return true;
    }

    return false;
}

void store_lds_cache(float3 position, float distance) {
    int8_t3 p = discretize_position(position);
    int8_t d = (int8_t)clamp(floor(distance * 4), 0, 255);
    int8_t4 l = int8_t4(p, d);

    GroupMemoryBarrierWithGroupSync();

    lds_data[index_hash(p)] = packInt4x8((int16_t4)l);
    lds_data2[index_hash(p)] = distance;

    GroupMemoryBarrierWithGroupSync();
}


TraceRes trace(float3 ray_pos, float3 ray_dir) {
#ifdef HWRT_BOUNDS_ACCEL
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> sq;
    let flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

    RayDesc desc;		

    desc.Origin = ray_pos;
    desc.Direction = ray_dir;
    desc.TMax = MAX_DISTANCE;
    desc.TMin = 0.0;

    if (all(thread_id == 0)) {
        lds_data = {};
        GroupMemoryBarrierWithGroupSync();
    }

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
        DifferentialPair<float3> dp_a = diffPair(point);
        
        bwd_diff(d)(dp_a, 1.0);  
        float3 normal = normalize(dp_a.d);
        return TraceRes(i, true, point, ray_dir, normal, 1.0, sdf_calls);
    } else {
        return TraceRes(i, false, ray_pos, ray_dir, 0.0, 1.0, sdf_calls);
    }
#else
    #define SDF_SOFT_SHADOWS_SUPPORTED
    float shadow = 1.0;
    float3 point = ray_pos;
    int i = 0;
    float distance = 0.0;

    // neat trick: threshold increase proportionally to distance AND pixel "factor" which takes partial derivative of ray dir (to give a sense of how "large" a ray is)
    // float pixel_thing = max(length(ddx(ray_dir)), length(ddy(ray_dir)));
    for (; i < MAX_ITERS; i++) {
        let k = d(point);

        if (k < THRESHOLD_DISTANCE) {
            // if (k < max(THRESHOLD_DISTANCE * distance * pixel_thing * 1000.0, THRESHOLD_DISTANCE)) {
            DifferentialPair<float3> dp_a = diffPair(point);
            
            bwd_diff(d)(dp_a, 1.0);  
            float3 normal = normalize(dp_a.d);
            return TraceRes(i, true, point, ray_dir, normal, 0.0, i);
        }

        if (k > MAX_DISTANCE) {
            break;
        }

        // https://iquilezles.org/articles/rmshadows/
        shadow = min(shadow, k * 8.0 / distance);
        distance += k;
        point += ray_dir * k * 0.90;
    }

    return TraceRes(i, false, ray_pos, ray_dir, 0.0, shadow, i);
#endif
}

#endif