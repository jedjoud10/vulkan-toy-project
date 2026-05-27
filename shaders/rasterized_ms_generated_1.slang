#language slang 2026
#include "utils/pbr.slang"
#include "utils/uniform_thing.slang"
#include "utils/bindless.slang"

struct VertexOutput {
    float3 world_pos;
    float3 meshlet_colour;
    float4 sv_position : SV_Position;
};

float density(uint3 pos, uint3 gid, int3 chunk_offset) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float3 p = (float3)pos + gid * MESHLET_SIZE + chunk_offset * (int3)CHUNK_SIZE /* + floor(uniforms.position.xyz) */; 

    float density = p.y - (2 + snoise(p * 0.02 + uniforms.time * 0.01 * 0) * 32);

    float sphere = length(p - uniforms.position.xyz) - 10;
    density = max(density, -sphere);
    return density;
}

const static uint MESHLET_SIZE = 4;
const static uint CHUNK_SIZE = MESHLET_SIZE * 8;

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

const static uint3[] EDGE_POSITIONS_0 = {
    uint3(0, 0, 0),
    uint3(1, 0, 0),
    uint3(1, 1, 0),
    uint3(0, 1, 0),
    uint3(0, 0, 1),
    uint3(1, 0, 1),
    uint3(1, 1, 1),
    uint3(0, 1, 1),
    uint3(0, 0, 0),
    uint3(1, 0, 0),
    uint3(1, 1, 0),
    uint3(0, 1, 0),
};

const static uint3[] EDGE_POSITIONS_1 = {
    uint3(1, 0, 0),
    uint3(1, 1, 0),
    uint3(0, 1, 0),
    uint3(0, 0, 0),
    uint3(1, 0, 1),
    uint3(1, 1, 1),
    uint3(0, 1, 1),
    uint3(0, 0, 1),
    uint3(0, 0, 1),
    uint3(1, 0, 1),
    uint3(1, 1, 1),
    uint3(0, 1, 1),
};

float unlerp(float x, float a, float b) {
    return (x - a) / (b - a);
}

// 1 element padding
const static int MESHLET_SIZE_PLUS_ONE = MESHLET_SIZE+1;
const static int MESHLET_VOLUME_PLUS_ONE = MESHLET_SIZE_PLUS_ONE*MESHLET_SIZE_PLUS_ONE*MESHLET_SIZE_PLUS_ONE;

// 2 element padding
const static int MESHLET_SIZE_PLUS_TWO = MESHLET_SIZE+2;
const static int MESHLET_VOLUME_PLUS_TWO = MESHLET_SIZE_PLUS_TWO*MESHLET_SIZE_PLUS_TWO*MESHLET_SIZE_PLUS_TWO;

groupshared float[MESHLET_VOLUME_PLUS_TWO] cached_density;
groupshared bool[MESHLET_VOLUME_PLUS_TWO] cached_is_set;
groupshared uint[MESHLET_VOLUME_PLUS_ONE] lookup;
groupshared Atomic<uint> vertex_count;
groupshared Atomic<uint> triangle_count;

struct DummyPayload {
    uint what;
}


// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("task")]
[numthreads(1, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    DummyPayload payload = DummyPayload(0);
    DispatchMesh(8, 8, 8, payload);
}

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(6, 6, 6)]
void mesh_main(out vertices VertexOutput vertices[128], out indices uint3 triangles[256], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uniform int3 chunk_offset) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    
    // initialize group shared memory
    if (all(gtid == 0)) {
        lookup = {};
        vertex_count = 0;
        triangle_count = 0;
    }

    GroupMemoryBarrierWithGroupSync();
    
    // port of https://github.com/jedjoud10/vulkan-toy-project/blob/main/src/voxel/meshes.rs
    // generate vertices
    uint3 cell = gtid;

    // cache density function and booleans
    float cell_density = density(cell, gid, chunk_offset);
    bool cell_set = cell_density < 0;
    cached_is_set[offset_to_index(cell, MESHLET_SIZE_PLUS_TWO)] = cell_set;
    cached_density[offset_to_index(cell, MESHLET_SIZE_PLUS_TWO)] = cell_density;
    
    GroupMemoryBarrierWithGroupSync();
    
    // create single vertex per thread
    if (all(gtid < MESHLET_SIZE_PLUS_ONE)) {
        /*
        uint8_t bitmask = 0;
        for (int neighbour = 0; neighbour < 8; neighbour++) {
            uint3 neighbour_offset = index_to_offset(neighbour, 2);

            if (cached_is_set[offset_to_index(cell + neighbour_offset, MESHLET_SIZE_PLUS_TWO)]) {
                bitmask |= 1 << neighbour;
            }
        }
        */

        float3 avg = 0;
        int count = 0;

        for (int edge = 0; edge < 12; edge++) {
            uint3 p1_offsetted = cell + EDGE_POSITIONS_0[edge];
            uint3 p2_offsetted = cell + EDGE_POSITIONS_1[edge];

            uint p1_offsetted_index = offset_to_index(p1_offsetted, MESHLET_SIZE_PLUS_TWO);
            uint p2_offsetted_index = offset_to_index(p2_offsetted, MESHLET_SIZE_PLUS_TWO);
            
            float d1 = cached_density[p1_offsetted_index];
            float d2 = cached_density[p2_offsetted_index];
            
            if (d1 < 0 ^ d2 < 0) {
                float factor = unlerp(0, d1, d2);
                avg += lerp(p1_offsetted, p2_offsetted, factor);
                count++;
            }
        }

        if (count > 0) {
            let vertex_index = vertex_count.increment(MemoryOrder.Relaxed);
            lookup[offset_to_index(cell, MESHLET_SIZE_PLUS_ONE)] = vertex_index;

            float3 loaded_vertex = avg / count /* + floor(uniforms.position.xyz) */ + gid * MESHLET_SIZE + (int3)CHUNK_SIZE * chunk_offset;

            float4 world_position = float4(loaded_vertex, 1);
            float4 t = mul(world_position, uniforms.view_projection_matrix);
            t.y = -t.y;
            t.x = -t.x;
            vertices[vertex_index].world_pos = world_position.xyz;
            vertices[vertex_index].sv_position = t;
            vertices[vertex_index].meshlet_colour = hash33(gid);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // create triangles (up to 3 quads) per thread 
    if (all(gtid < MESHLET_SIZE_PLUS_ONE)) {
        

        // generate triangles    
        // TODO: optimize this by only visiting the edges that were previously handled in the vertex stage (pass a 3 bit bitmask for example)
        for (int axis = 0; axis < 3; axis++) {
            if (cell[axis] == MESHLET_SIZE) {
                continue;
            }
        
            uint3 next_cell = cell;
            next_cell[axis] += 1;
            
            if (cell_set != cached_is_set[offset_to_index(next_cell, MESHLET_SIZE_PLUS_TWO)]) {
                let dir = cell_set ^ (axis == 1);

                

                uint4 quad_vertex_indices = 676767;
                uint3[4] vertex_offsets = quad_vertex_offsets_for_axis(axis);
            
                for (int index = 0; index < 4; index++) {
                    let target = vertex_offsets[index] + next_cell - 1;
                
                    if (all(target >= 0 & target < MESHLET_SIZE_PLUS_ONE)) {
                        let vertex_offset = offset_to_index(target, MESHLET_SIZE_PLUS_ONE);
                        let vertex_index = lookup[vertex_offset];
                        quad_vertex_indices[index] = vertex_index;
                    }
                }
            
                if (any(quad_vertex_indices == 676767)) {
                    continue;
                }
            
                //const static uint[6] INDEX_QUAD_ORDER = {0, 1, 2, 2, 1, 3};
                //const static uint[6] INDEX_OPPOSITE_QUAD_ORDER = {1, 0, 2, 1, 2, 3};

                let triangle_index = triangle_count.add(2, MemoryOrder.Relaxed);
                if (dir) {
                    triangles[triangle_index] = uint3(quad_vertex_indices.x, quad_vertex_indices.y, quad_vertex_indices.z);
                    triangles[triangle_index+1] = uint3(quad_vertex_indices.z, quad_vertex_indices.y, quad_vertex_indices.w);
                } else {
                    triangles[triangle_index] = uint3(quad_vertex_indices.y, quad_vertex_indices.x, quad_vertex_indices.z);
                    triangles[triangle_index+1] = uint3(quad_vertex_indices.y, quad_vertex_indices.z, quad_vertex_indices.w);
                }
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    SetMeshOutputCounts(vertex_count.load(MemoryOrder.Relaxed), triangle_count.load(MemoryOrder.Relaxed));
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let skybox_sampler = sampled_images_cube[SKYBOX_SAMPLER_IMAGE_IDX];
    let clouds_sampler = sampled_images[CLOUDS_SAMPLER_IMAGE_IDX];
    
    float3 world = in_vert.world_pos;
    
    //float3 normal = normalize(in_vert.normal + float3(snoise(world.xz * 0.124) * 0.01));
    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));
    float3 view = normalize(in_vert.world_pos - uniform_buffer.position.xyz);

    float3 refraction = sample_skybox_and_clouds(world, reflect(view, normal), clouds_sampler, skybox_sampler);

    if (uniform_buffer.debug_type % 2 == 0) {
        return float4(refraction, 1);
    } else {
        return float4(in_vert.meshlet_colour.xyz, 1);
    }
}