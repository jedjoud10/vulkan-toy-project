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

    let planar_xy = image.Sample(sampler, world.xy * scale).xyz;
    let planar_xz = image.Sample(sampler, world.xz * scale).xyz;
    let planar_zy = image.Sample(sampler, world.zy * scale).xyz;

    return planar_xy * blending.z + planar_xz * blending.y + planar_zy * blending.x;
}


// https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a
// https://github.com/jedjoud10/cflake-engine/blob/main/crates/assets/src/assets/engine/shaders/scene/terrain/terrain.frag#L94-L108
float3 sample_triplanar_normal(uint index, float3 world, float3 normal) {
    let image = sampled_images[index];
    let sampler = samplers[LINEAR_UNCAMPLED_SAMPLER_IDX];
    let scale = TRIPLANAR_SCALE;

    float3 planar_xy = image.Sample(sampler, world.xy * scale).xyz * 2.0 - 1.0;
    float3 planar_xz = image.Sample(sampler, world.xz * scale).xyz * 2.0 - 1.0;
    float3 planar_zy = image.Sample(sampler, world.zy * scale).xyz * 2.0 - 1.0;

    planar_xy.y *= -1;
    planar_xz.y *= -1;
    planar_zy.y *= -1;

    float3 blending = abs(normal);
    blending = pow(blending, TRIPLANAR_WEIGHT_POW);
    blending = normalize(blending);

    let normal_strength = TRIPLANAR_NORMAL_STRENGTH;
    float3 normalx = float3(planar_zy.xy * normal_strength + normal.zy, normal.x) * blending.x;
	float3 normaly = float3(planar_xz.xy * normal_strength + normal.xz, normal.y) * blending.y;
	float3 normalz = float3(planar_xy.xy * normal_strength + normal.xy, normal.z) * blending.z;
	return normalize(normalx.zyx + normaly.xzy + normalz.xyz);
}

#endif