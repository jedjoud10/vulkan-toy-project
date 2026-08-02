#ifndef SDF
#define SDF

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"


// https://iquilezles.org/articles/distfunctions/
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

[Differentiable]
float3x3 AngleAxis3x3(float angle, float3 axis)
{
    float c, s;
    sincos(angle, s, c);

    float t = 1 - c;
    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    return float3x3(
        t * x * x + c,      t * x * y - s * z,  t * x * z + s * y,
        t * x * y + s * z,  t * y * y + c,      t * y * z - s * x,
        t * x * z - s * y,  t * y * z + s * x,  t * z * z + c
    );
}


[BackwardDerivativeOf(noise)]
void bwd_test(inout DifferentialPair<float3> p_pair, float dOut)
{
    float eps = 0.001;
    float x = noise(p_pair.p - float3(eps, 0, 0));
    float y = noise(p_pair.p - float3(0, eps, 0));
    float z = noise(p_pair.p - float3(0, 0, eps));
    float b = noise(p_pair.p);


    p_pair = diffPair<float3>(p_pair.p, float3(b-x, b-y, b-z));
}

[Differentiable]
float d(float3 p, float t) {
    // float s = length(p - 10*clamp(round(p/10.0), -float3(100, 1, 100), float3(100, 1, 100))) - 2;
    float sphere = length(p + float3(0, -1, 0)) - 2;
    float octa = sdOctahedron(p - float3(0, 20, 0), 2.0);
    
    float a = sdHexPrism(p + float3(-10, -2, 0,), float2(2, 4));
    float b = sdHexPrism(p + float3(-10, -2, 0,), float2(1, 5));

    float hex = max(-b, a);
    float cyl = sdCylinder(p + float3(10, 0, 0), float3(0, 0, 2));

    float3x3 rot = AngleAxis3x3(2.4, normalize(float3(1, 1, 1)));
    float tor = sdTorus(mul(p, rot) + float3(-3, -5, 3), float2(5, 2)) + noise(p * 5.0) * 0.02;
    
    return min(sphere, min(p.y, min(min(octa, cyl), min(hex, tor))));
    

    
    /*
    // TODO: fix this lolz
    uint count = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<uint32_t>(0);
    float k = 1000;
    [MaxIters(1)]
    for (int element = 0; element < count; element++) {
        float3 origin = storage_buffers_readonly[STORAGE_BUFFER_SCENE_REPR_BUFFER].Load<float3>(4 + element * 12);
        float deformation = (sin(p.y * 20)+sin(p.z * 20)+sin(p.x * 20)) * 0.01;
        k = min(k, length(origin-p) - 1 + deformation);

        if (k < 0) {
            return k;
        }
    }
    return k;
    */

    // return length(p) - 1.5;
}

struct TraceRes {
    int iters;
    bool hit;
    float3 pos;
    float3 dir;
    float3 normal;
    float shadow;
};

static const uint MAX_ITERS = 256;
static const float MAX_DISTANCE = 1000;
static const float THRESHOLD_DISTANCE = 0.001;

float ray_box_get_tmax(
    float3 ray_pos,
    float3 ray_dir,
    float3 aabb_min,
    float3 aabb_max,
) {
    // TODO: optimize if required
    float3 t1 = (aabb_min - ray_pos) / ray_dir;
    float3 t2 = (aabb_max - ray_pos) / ray_dir;

    // float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    // float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);
    return tmax;
    // return max(tmin, 0) <= tmax;
}

bool trace_inner<uint T>(inout RayQuery<T> sq, inout int i, inout float distance) {
    // please just for the love of everything please just give me candidate ray distance that works with procedural primitive... please...
    // not ONLY that but the fucking order of CANDIDATE_PROCEDURAL_PRIMITIVE is incorrect! it's not even fucking sorted!
    // distance = ray_box_intersection(sq.CandidateRayObjectRayOrigin(), sq.CandidateObjectRayDirection(), -1.0, +1.0);
    
    float tmax = ray_box_get_tmax(sq.CandidateRayObjectRayOrigin(), sq.CandidateObjectRayDirection(), -1.0, 1.0);
    
    float3 point = sq.WorldRayOrigin() + sq.WorldRayDirection() * distance;



    for (; i < MAX_ITERS; i++) {
        float k = d(point, 1.0);

        // check if we got close enough to the surface
        if (k < THRESHOLD_DISTANCE) {
            sq.CommitProceduralPrimitiveHit(distance);  // forcing to be opaque
            return true;
        }

        // check if we left the bounds of the AABB (tmax is calculated using software ray-AABB test)
        if (distance > tmax) {
            return false;
        }

        point += sq.WorldRayDirection() * k * 0.90;
        distance += k * 0.90;
    }

    return false;
}

TraceRes trace(float3 ray_pos, float3 ray_dir) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES> sq;
    let flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES;

    RayDesc desc;		

    desc.Origin = ray_pos;
    desc.Direction = ray_dir;
    desc.TMax = MAX_DISTANCE;
    desc.TMin = 0.001;

    sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);

    int i = 0;
    float distance = 0.0;
    while (sq.Proceed()) {
        if(sq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            if (trace_inner(sq, i, distance)) {
                sq.Abort();
            }

            if (i == MAX_ITERS || distance >= MAX_DISTANCE) {
                sq.Abort();
            }
        }
    }

    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        float3 point = sq.WorldRayOrigin() + sq.WorldRayDirection() * distance;
        // if (k < max(THRESHOLD_DISTANCE * distance * pixel_thing * 1000.0, THRESHOLD_DISTANCE)) {
        DifferentialPair<float3> dp_a = diffPair(point);
        DifferentialPair<float> dp_t = diffPair(uniforms.time);
        
        bwd_diff(d)(dp_a, dp_t, 1.0);  
        float3 normal = normalize(dp_a.d);
        return TraceRes(i, true, point, ray_dir, normal, 1.0);
    } else {
        return TraceRes(i, false, ray_pos, ray_dir, 0.0, 1.0);
    }
    
    /*
    float shadow = 1.0;
    float3 point = ray_pos;
    int i = 0;
    float distance = 0.0;

    // neat trick: threshold increase proportionally to distance AND pixel "factor" which takes partial derivative of ray dir (to give a sense of how "large" a ray is)
    // float pixel_thing = max(length(ddx(ray_dir)), length(ddy(ray_dir)));
    for (; i < MAX_ITERS; i++) {
        let k = d(point, uniforms.time);

        if (k < THRESHOLD_DISTANCE) {
            // if (k < max(THRESHOLD_DISTANCE * distance * pixel_thing * 1000.0, THRESHOLD_DISTANCE)) {
            DifferentialPair<float3> dp_a = diffPair(point);
            DifferentialPair<float> dp_t = diffPair(uniforms.time);
            
            bwd_diff(d)(dp_a, dp_t, 1.0);  
            float3 normal = normalize(dp_a.d);
            return TraceRes(i, true, point, ray_dir, normal, 0.0);
        }

        if (k > MAX_DISTANCE) {
            break;
        }

        // https://iquilezles.org/articles/rmshadows/
        shadow = min(shadow, k * 8.0 / distance);
        distance += k;
        point += ray_dir * k * 0.90;
    }

    return TraceRes(i, false, ray_pos, ray_dir, 0.0, shadow);
    */
}

#endif