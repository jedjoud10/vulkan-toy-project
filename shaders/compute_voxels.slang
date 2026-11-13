#include "utils/bindless.slang"
#include "utils/noises.slang"

struct PushConstants {
    float4 chunk_position_and_size;
}

[shader("compute")]
[numthreads(4,4,4)]
void main(uint3 id: SV_DispatchThreadID, uniform PushConstants pc) {
    let texture = storage_images_3d_float[VOXEL_DENSITY_STORAGE_IMAGE_IDX];
    let p = float3(id) * pc.chunk_position_and_size.w + pc.chunk_position_and_size.xyz;

    float d = snoise(p * 0.01 * float3(1,2,1) + 423.2342) * 10 + p.y;
    d += snoise(p * 0.05 + snoise(p * 0.2) * 0.0) * 5;

    texture.Store(id, d);
}