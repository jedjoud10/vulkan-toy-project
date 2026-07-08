#ifndef TRIPLANAR
#define TRIPLANAR
#include "bindless.slang"

float3 sample_triplanar(uint index, float3 world, float3 normal) {
    let image = sampled_images[index];
    let sampler = samplers[NEAREST_SAMPLER_IDX];
    let scale = 0.2;

    let planar_xy = image.Sample(sampler, world.xy * scale).xyz;
    let planar_xz = image.Sample(sampler, world.xz * scale).xyz;
    let planar_zy = image.Sample(sampler, world.zy * scale).xyz;

    let abs_normal = abs(normal);

    return planar_xy * abs_normal.z + planar_xz * abs_normal.y + planar_zy * abs_normal.x;
}


// https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a
// https://github.com/jedjoud10/cflake-engine/blob/main/crates/assets/src/assets/engine/shaders/scene/terrain/terrain.frag#L94-L108
float3 sample_triplanar_normal(uint index, float3 world, float3 normal) {
    let image = sampled_images[index];
    let sampler = samplers[NEAREST_SAMPLER_IDX];
    let scale = 0.2;

    let planar_xy = image.Sample(sampler, world.xy * scale).xyz * 2.0 - 1.0;
    let planar_xz = image.Sample(sampler, world.xz * scale).xyz * 2.0 - 1.0;
    let planar_zy = image.Sample(sampler, world.zy * scale).xyz * 2.0 - 1.0;

    let blending = abs(normal);

    let normal_strength = 2.0;
    let normalx = float3(float2(planar_zy.x, planar_zy.y) * normal_strength + normal.zy, normal.x) * blending.x;
	let normaly = float3(float2(planar_xz.x, planar_xz.y) * normal_strength + normal.xz, normal.y) * blending.y;
	let normalz = float3(float2(planar_xy.x, planar_xy.y) * normal_strength + normal.xy, normal.z) * blending.z;
	return normalize(normalx.zyx + normaly.xzy + normalz.xyz);
}

#endif