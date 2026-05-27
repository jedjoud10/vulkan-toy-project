#language slang 2026
#include "utils/pbr.slang"
#include "utils/uniform_thing.slang"
#include "utils/bindless.slang"

struct VertexOutput {
    float3 world_pos;
    float3 meshlet_colour;
    float4 sv_position : SV_Position;
};

bool is_set(uint3 pos) {
    return pos.y < 2;
}

uint3 index_to_offset(int index, int size) {
    let x = index % size;
    let y = (index / size) % size;
    let z = index / (size*size);
    return uint3(x,y,z);
}

int offset_to_index(uint3 offset, int size) {
    return  offset.x + offset.y * size + offset.z * size * size;
}

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[128], out indices uint3 triangles[128], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    uint[64] lookup = {};
    uint vertex_count = 0;

    // generate vertices
    for (int x = 0; x < 4; x++) {
        for (int y = 0; y < 4; y++) {
            for (int z = 0; z < 4; z++) {
                uint3 cell = uint3(x,y,z);

                uint8_t bitmask = 0;
                for (int neighbour = 0; neighbour < 8; neighbour++) {
                    uint3 neighbour_offset = index_to_offset(neighbour, 2);

                    if (is_set(cell + neighbour_offset)) {
                        bitmask |= 1 << neighbour;
                    }
                }

                if (bitmask != 0 && bitmask != 0xFF) {
                    lookup[offset_to_index(cell, 4)] = vertex_count;

                    float3 loaded_vertex = cell;

                    float4 world_position = float4(loaded_vertex, 1);
                    float4 t = mul(world_position, uniforms.view_projection_matrix);
                    t.y = -t.y;
                    t.x = -t.x;
                    vertices[vertex_count].world_pos = world_position.xyz;
                    vertices[vertex_count].sv_position = t;
                    vertices[vertex_count].meshlet_colour = hash33(gid);

                    vertex_count++;
                }
            }
        }
    }

    // generate triangles
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    return float4(in_vert.meshlet_colour, 1);
}