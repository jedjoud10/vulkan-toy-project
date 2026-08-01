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

[Differentiable]
float d(float3 p, float t) {
    // float s = length(p - 10*clamp(round(p/10.0), -float3(100, 1, 100), float3(100, 1, 100))) - 2;
    float sphere = length(p + float3(0, -1, 0)) - 2;
    float octa = sdOctahedron(p - float3(0, 20, 0), 2.0);
    float hex = sdHexPrism(p - float3(10, 0, 0,), float2(3, 4));
    return min(sphere, min(p.y, min(octa, hex)));
}

struct TraceRes {
    int iters;
    bool hit;
    float3 pos;
    float3 dir;
    float3 diff;
    float closest;
};

static const uint MAX_ITERS = 256;
static const float MAX_DISTANCE = 1000;
static const float THRESHOLD_DISTANCE = 0.001;


TraceRes trace(float3 ray_pos, float3 ray_dir) {
    let uniforms = load_per_frame_uniforms();
    float closest = 1.0;
    float3 point = ray_pos;
    int i = 0;
    float distance = 0.0;
    for (; i < MAX_ITERS; i++) {
        let k = d(point, uniforms.time);

        if (k < THRESHOLD_DISTANCE) {
            DifferentialPair<float3> dp_a = diffPair(point);
            DifferentialPair<float> dp_t = diffPair(uniforms.time);
            
            bwd_diff(d)(dp_a, dp_t, 1.0);  
            float3 normal = normalize(dp_a.d);
            return TraceRes(i, true, point, ray_dir, normal, closest);
        }

        if (k > MAX_DISTANCE) {
            break;
        }

        distance += k;
        point += ray_dir * k;
        closest = min(k, closest);
    }

    return TraceRes(i, false, ray_pos, ray_dir, 0.0, closest);
}

#endif