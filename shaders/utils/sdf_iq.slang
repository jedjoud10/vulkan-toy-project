#ifndef SDF_IQ
#define SDF_IQ

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"


typedef int3 ivec3 

// The MIT License
// Copyright © 2019 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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


// https://iquilezles.org/articles/smin
float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}

// https://iquilezles.org/articles/fbmsdf/
float sph( ivec3 i, vec3 f, ivec3 c )
{
   // random radius at grid vertex i+c
   float rad = 0.5*hash13(i+c);
   // distance to sphere at grid vertex i+c
   return length(f-vec3(c)) - rad; 
}

// https://iquilezles.org/articles/fbmsdf/
float sdBase( vec3 p )
{
   ivec3 i = ivec3(floor(p));
    vec3 f =       fract(p);
   // distance to the 8 corners spheres
   return min(min(min(sph(i,f,ivec3(0,0,0)),
                      sph(i,f,ivec3(0,0,1))),
                  min(sph(i,f,ivec3(0,1,0)),
                      sph(i,f,ivec3(0,1,1)))),
              min(min(sph(i,f,ivec3(1,0,0)),
                      sph(i,f,ivec3(1,0,1))),
                  min(sph(i,f,ivec3(1,1,0)),
                      sph(i,f,ivec3(1,1,1)))));
}

const static float3x3 matty = float3x3( 0.00, 1.60, 1.20, -1.60, 0.72,-0.96, -1.20,-0.96, 1.28 );

// https://iquilezles.org/articles/fbmsdf/
float sdFbm( vec3 p, float d )
{
   float s = 1.0;
   for( int i=0; i<4; i++ )
   {
       // evaluate new octave
       float n = s*sdBase(p);
	
       // add
       n = smax(n,d-0.15*s,0.001*s);
       d = smin(n,d      ,0.001*s);
	
       // prepare next octave
       p = mul(p, matty);
       s = 0.22*s;
   }
   return d;
}


const static int TEXTURE_SIZE = 128;
float tex_sample(float3 p) {
    return sampled_images_3d[SDF_TEXTURE_SAMPLED_IMAGE_IDX].Sample(samplers[LINEAR_UNCAMPLED_SAMPLER_IDX], p / TEXTURE_SIZE + 0.5).x;
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

#endif