#include "utils/bindless.slang"
#include "utils/noises.slang"

uint3 index_to_offset(uint index, uint size) {
    let x: uint = index % size;
    let y: uint = (index / size) % size;
    let z : uint = index / (size * size);
    return uint3(x, y, z);
}

uint offset_to_index(uint3 offset, uint size) {
    return offset.x + offset.y * size + offset.z * size * size;
}

struct PushConstants {
    float4 *chunk_positions_and_sizes;
    uint num_chunks;
    uint _padding;
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

    // FIXME: add constants
    let chunk_offset : uint3 = id / 34;
    let local_id : uint3 = id % 34;

    // convert chunk offset to contiguous chunk index [0..64)
    let chunk_index: uint = offset_to_index(chunk_offset, 4);

    if (chunk_index >= pc.num_chunks) {
        return;
    }

    let chunk_position_and_size: float4 = pc.chunk_positions_and_sizes[chunk_index];

    float3 p = float3(local_id) * chunk_position_and_size.w + chunk_position_and_size.xyz;
    p *= 0.05;

    float d = fbm(p * 0.01 * float3(1,3,1));
    d = smoothstep(-1, .3, d) * 10;
    // d = min(d, snoise(p * 0.01 * float3(1,2,1)) * 50);

    d += p.y - 10;

    d += -cellular(p * 1.5).x * 0.1;
    d += smoothstep(-0.1, 0.1, snoise(p * 1.5 * float3(1, 3, 1))) * 0.05;

    // float d = snoise(p * 0.01 * float3(1,2,1) + 423.2342) * 10 + p.y;
    // d += snoise(p * 0.05 + snoise(p * 0.2) * 0.0) * 5;

    texture.Store(id, d);
}