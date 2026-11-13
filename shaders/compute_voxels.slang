#include "utils/bindless.slang"
#include "utils/noises.slang"

struct PushConstants {
    float4 chunk_position_and_size;
}

float fbm(float3 world) {
    float amplitude = 1;
    float result = 0;
	for (int i = 0; i < 5; i++) {
        world *= 2.5;
        amplitude *= 0.4;

        float l = snoise(world) * amplitude;
        // l = smoothstep(-1, 0, l) * amplitude;

        result += l;
	}

    return result;
}

[shader("compute")]
[numthreads(4,4,4)]
void main(uint3 id: SV_DispatchThreadID, uniform PushConstants pc) {
    let texture = storage_images_3d_float[VOXEL_DENSITY_STORAGE_IMAGE_IDX];
    float3 p = float3(id) * pc.chunk_position_and_size.w + pc.chunk_position_and_size.xyz;
    p *= 0.2;

    float d = fbm(p * 0.01 * float3(1,3,1));
    d = smoothstep(-1, .3, d) * 10;
    // d = min(d, snoise(p * 0.01 * float3(1,2,1)) * 50);

    d += p.y;
    // float d = snoise(p * 0.01 * float3(1,2,1) + 423.2342) * 10 + p.y;
    // d += snoise(p * 0.05 + snoise(p * 0.2) * 0.0) * 5;

    texture.Store(id, d);
}