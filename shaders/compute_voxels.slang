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

        float3 octave_offset = hash33(float3(i) * float3(12.345, 40.32, 50.543)) * 1000.0;

        float l = snoise(world + octave_offset) * amplitude;
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

    float3 warp_scale = float3(1, 3, 1) * 0.001;
    float3 warp = float3(snoise(p * warp_scale), snoise(p * warp_scale + 43.23), snoise(p * warp_scale - 54.43435)) * 0.1;

    float d = fbm(p * 0.0002 * float3(1,0,1) + warp);
    d = smoothstep(-1, .8, d) * 200;

    d += p.y - 150;

    // d += srdnoise(p.xz, 0).x * 1;

    // float d = snoise(p * 0.01 * float3(1,2,1) + 423.2342) * 10 + p.y;
    // d += snoise(p * 0.05 + snoise(p * 0.2) * 0.0) * 5;

    texture.Store(id, d);
}