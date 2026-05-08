#language slang 2026

[[vk::binding(0, 0)]]
RWTexture3D<uint8_t> sparse_voxel_texture;

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 id: SV_DispatchThreadID, uniform uint4 chunk_position) {
}
