#language slang 2026
#include "utils/bindless.slang"

typedef int3 ivec3 
typedef float3 vec3 

// The MIT License
// Copyright © 2019 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// cubic polynomial
// https://iquilezles.org/articles/smin
float smin( float a, float b, float k )
{
    k *= 6.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*h*k*(1.0/6.0);
}

// https://iquilezles.org/articles/smin
float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}

float sph( ivec3 i, vec3 f, ivec3 c )
{
   // random radius at grid vertex i+c
   float rad = 0.5*hash13(i+c);
   // distance to sphere at grid vertex i+c
   return length(f-vec3(c)) - rad; 
}

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
       n = smax(n,d-0.15*s,0.1*s);
       d = smin(n,d      ,0.1*s);
	
       // prepare next octave
       p = mul(p, matty);
       s = 0.22*s;
   }
   return d;
}


[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id: SV_DispatchThreadID) {
    float scale = 4.0;
    float3 p = id.xyz * 0.01;
    let uniforms = load_per_frame_uniforms();
    float d = sdFbm(p + float3(uniforms.time * 0.01, 0, 0), p.y - 0.6) * 100.0;

    //float d = sdBase(p);
    //d = p.y - 64;
    storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Store(id.xyz, d);
}

struct aabb {
    float3 min;
    float3 max;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void compute_aabbs(uint3 id: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    float loaded = storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Load(id.xyz).x;

    float dmin = WaveActiveMin(loaded);
    float dmax = WaveActiveMax(loaded);

    float3 k = id.xyz - 64.0;

    float3 aabb_min = 5;
    float3 aabb_max = 0;

    bool close_enough = loaded < 2.5;

    float3 t = gtid;

    // Reduce each component individually
    aabb_min.x = WaveActiveMin(select(close_enough, t.x, 0));
    aabb_min.y = WaveActiveMin(select(close_enough, t.y, 0));
    aabb_min.z = WaveActiveMin(select(close_enough, t.z, 0));

    aabb_max.x = WaveActiveMax(select(close_enough, t.x+1, 0));
    aabb_max.y = WaveActiveMax(select(close_enough, t.y+1, 0));
    aabb_max.z = WaveActiveMax(select(close_enough, t.z+1, 0));
    int3 lid2 = (int3)gid - 16;

    uint aaabb_byte_offset = 3*4*2;

    GroupMemoryBarrierWithWaveSync();

    if (dmax > -1 && dmin < 2.5 && all(gtid == 0)) {
        uint count = 0;
    
        storage_buffers[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS+1].InterlockedAdd(0, 1, count);
        
        if (count < 1024*1024*1024) {
            float offset = 1;
            aabb data = aabb(aabb_min + gid * 4 - 64.0 + offset, aabb_max + gid * 4 - 64.0 + offset);
            // aabb data = aabb(gid * 4 - 64.0, gid * 4 + 4 - 64.0);

            storage_buffers[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS].Store(aaabb_byte_offset * count, data.min);
            storage_buffers[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS].Store(aaabb_byte_offset * count + 3*4, data.max);
        }
    }
    
    GroupMemoryBarrierWithWaveSync();
}