#language slang 2026
#include "utils/pbr.slang"
#include "utils/uniform_thing.slang"
#include "utils/bindless.slang"

struct VertexOutput {
    float3 world_pos;
    float3 meshlet_colour;
    float4 sv_position : SV_Position;
};

float density(uint3 pos, uint3 gid) {
    float3 p = (float3)pos + gid * (CHUNK_SIZE-1); 
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    return p.y - (2 + snoise(p * 0.02 + uniforms.time * 0.01) * 6);
}


bool is_set(uint3 pos, uint3 gid) {
    return density(pos, gid) < 0;
}

const static uint CHUNK_SIZE = 4;

uint3 index_to_offset(int index, int size) {
    let x = index % size;
    let y = (index / size) % size;
    let z = index / (size*size);
    return uint3(x,y,z);
}

int offset_to_index(uint3 offset, int size) {
    return  offset.x + offset.y * size + offset.z * size * size;
}


uint3[4] quad_vertex_offsets_for_axis(uint axis) {
    switch (axis) {
        case 0: // x
            return {uint3(0, 0, 0), uint3(0, 1, 0), uint3(0, 0, 1), uint3(0, 1, 1)};
        case 1: // y
            return {uint3(0, 0, 0), uint3(1, 0, 0), uint3(0, 0, 1), uint3(1, 0, 1)}; 
        case 2: // z
            return {uint3(0, 0, 0), uint3(1, 0, 0), uint3(0, 1, 0), uint3(1, 1, 0)}; 
        default:
            return {};
    }
    
    return {};
}

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[128], out indices uint3 triangles[128], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    uint[64] lookup = {};
    uint vertex_count = 0;

    // port of https://github.com/jedjoud10/vulkan-toy-project/blob/main/src/voxel/meshes.rs
    

    // generate vertices
    for (int x = 0; x < CHUNK_SIZE-1; x++) {
        for (int y = 0; y < CHUNK_SIZE-1; y++) {
            for (int z = 0; z < CHUNK_SIZE-1; z++) {
                uint3 cell = uint3(x,y,z);

                uint8_t bitmask = 0;
                for (int neighbour = 0; neighbour < 8; neighbour++) {
                    uint3 neighbour_offset = index_to_offset(neighbour, 2);

                    if (is_set(cell + neighbour_offset, gid)) {
                        bitmask |= 1 << neighbour;
                    }
                }

                if (bitmask != 0 && bitmask != 0xFF) {
                    lookup[offset_to_index(cell, CHUNK_SIZE)] = vertex_count;

                    float3 loaded_vertex = cell + float3(10, 0, 0) + gid * (CHUNK_SIZE-2);

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

    int triangle_count = 0;

    // generate triangles
    for (int x = 0; x < CHUNK_SIZE-1; x++) {
        for (int y = 0; y < CHUNK_SIZE-1; y++) {
            for (int z = 0; z < CHUNK_SIZE-1; z++) {
                uint3 cell = uint3(x,y,z);
                bool cell_set = is_set(cell, gid);
                
                for (int axis = 0; axis < 3; axis++) {
                    uint3 next_cell = cell;
                    next_cell[axis] += 1;

                    if (cell_set != is_set(next_cell, gid)) {
                        let dir = cell_set ^ (axis == 1);

                        uint4 quad_vertex_indices = 676767;
                        uint3[4] vertex_offsets = quad_vertex_offsets_for_axis(axis);

                        for (int index = 0; index < 4; index++) {
                            let target = vertex_offsets[index] + next_cell - 1;

                            if (all(target >= 0 & target < CHUNK_SIZE)) {
                                let vertex_offset = offset_to_index(target, CHUNK_SIZE);
                                let vertex_index = lookup[vertex_offset];
                                quad_vertex_indices[index] = vertex_index;
                            }
                        }

                        if (any(quad_vertex_indices == 676767)) {
                            continue;
                        }

                        //const static uint[6] INDEX_QUAD_ORDER = {0, 1, 2, 2, 1, 3};
                        //const static uint[6] INDEX_OPPOSITE_QUAD_ORDER = {1, 0, 2, 1, 2, 3};
                        if (dir) {
                            triangles[triangle_count] = uint3(quad_vertex_indices.x, quad_vertex_indices.y, quad_vertex_indices.z);
                            triangles[triangle_count+1] = uint3(quad_vertex_indices.z, quad_vertex_indices.y, quad_vertex_indices.w);
                        } else {
                            triangles[triangle_count] = uint3(quad_vertex_indices.y, quad_vertex_indices.x, quad_vertex_indices.z);
                            triangles[triangle_count+1] = uint3(quad_vertex_indices.y, quad_vertex_indices.z, quad_vertex_indices.w);
                        }
                        
                        triangle_count += 2;
                    }
                }
            }
        }
    }

    SetMeshOutputCounts(vertex_count, triangle_count);
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    return float4(in_vert.meshlet_colour, 1);
}