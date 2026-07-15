#ifndef TRIPLANAR
#define TRIPLANAR
#include "bindless.slang"

static const float TRIPLANAR_SCALE = 0.2;
static const float TRIPLANAR_WEIGHT_POW = 2;
static const float TRIPLANAR_NORMAL_STRENGTH = 2.0;


// https://cprimozic.net/notes/posts/a-small-change-to-improve-triplanar-mapping/
float3 sample_triplanar(uint index, float3 world, float3 normal) {
    let image = sampled_images[index];
    let sampler = samplers[LINEAR_UNCAMPLED_SAMPLER_IDX];
    let scale = TRIPLANAR_SCALE;

    float3 blending = abs(normal);
    blending = pow(blending, TRIPLANAR_WEIGHT_POW);
    blending = normalize(blending);

    float3 output = 0.0;

    if (blending.x > 0.0) {
        output += image.Sample(sampler, world.xy * scale).xyz * blending.x;
    }

    if (blending.y > 0.0) {
        output += image.Sample(sampler, world.xz * scale).xyz * blending.y;
    }

    if (blending.z > 0.0) {
        output += image.Sample(sampler, world.zy * scale).xyz * blending.z;
    }

    return output;
}


// https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a
// https://github.com/jedjoud10/cflake-engine/blob/main/crates/assets/src/assets/engine/shaders/scene/terrain/terrain.frag#L94-L108
float3 sample_triplanar_normal(uint index, float3 world, float3 normal) {
    let image = sampled_images[index];
    let sampler = samplers[LINEAR_UNCAMPLED_SAMPLER_IDX];
    let scale = TRIPLANAR_SCALE;


    float3 blending = abs(normal);
    blending = pow(blending, TRIPLANAR_WEIGHT_POW);
    blending = normalize(blending);

    let normal_strength = TRIPLANAR_NORMAL_STRENGTH;

    float3 new_normal = 0.0;

    
    if (blending.x > 0.0) {
        float3 planar_zy = ((image.Sample(sampler, world.zy * scale).xyz * 2.0 - 1.0) * float3(1, -1, 1));
        float3 normalx = float3(planar_zy.xy * normal_strength + normal.zy, normal.x) * blending.x;
        new_normal += normalx.zyx;
    }
    
    if (blending.y > 0.0) {
        float3 planar_xz = ((image.Sample(sampler, world.xz * scale).xyz * 2.0 - 1.0) * float3(1, -1, 1));
        float3 normaly = float3(planar_xz.xy * normal_strength + normal.xz, normal.y) * blending.y;
        new_normal += normaly.xzy;
    }
    
    if (blending.z > 0.0) {
        float3 planar_xy = ((image.Sample(sampler, world.xy * scale).xyz * 2.0 - 1.0) * float3(1, -1, 1));
        float3 normalz = float3(planar_xy.xy * normal_strength + normal.xy, normal.z) * blending.z;
        new_normal += normalz.xyz;
    }



    
	return normalize(new_normal);
}

#endif