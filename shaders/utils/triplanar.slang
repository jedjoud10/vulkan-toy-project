#ifndef TRIPLANAR
#define TRIPLANAR
#include "bindless.slang"

float3 sample_triplanar_albedo(uint index, float3 world, float3 normal) {
    let image = sampled_images[index];
    let sampler = samplers[NEAREST_SAMPLER_IDX];
    let scale = 2.0;

    let planar_xy = image.Sample(sampler, world.xy * scale).xyz;
    let planar_xz = image.Sample(sampler, world.xz * scale).xyz;
    let planar_zy = image.Sample(sampler, world.zy * scale).xyz;

    let abs_normal = abs(normal);

    return planar_xy * abs_normal.z + planar_xz * abs_normal.y + planar_zy * abs_normal.x;
}

#endif