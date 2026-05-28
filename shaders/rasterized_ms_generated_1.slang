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
    let lights_buffer = storage_buffers_readonly[STORAGE_BUFFER_LIGHTS_BUFFER];
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float3 p = (float3)pos + gid * MESHLET_SIZE + chunk_offset * (int3)CHUNK_SIZE /* + floor(uniforms.position.xyz) */; 

    float density = 200;
    for (int i = 0; i < 500; i++) {
        float4 loaded = lights_buffer.Load<float4>(i * 4 * 4);
        float sphere = length(p - loaded.xyz) - 20;
        density = min(sphere, density);
    }

    return density;
    //return p.y + snoise(p * 0.05) * 10;

    /*
    float warp_scale = 0.2;
    float3 warp = float3(snoise(p*warp_scale + -2.213 + uniforms.time * 1.5), snoise(p*warp_scale + 4.12 -  uniforms.time * 1.01), snoise(p*warp_scale + 3.123 +  uniforms.time * 1.04));

    float density = p.y - (2 + snoise(p * 0.02 + warp * 0.0) * 32);

    float sphere = length(p - uniforms.position.xyz) - 10;
    density = max(density, -sphere);
    return density;
    */
}

const static uint MESHLET_SIZE = 4;
const static uint CHUNK_SIZE = MESHLET_SIZE * 4;

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

// Stolen from https://gist.github.com/dwilliamson/c041e3454a713e58baf6e4f8e5fffecd
const static uint16_t[] EDGE_MASKS = {
    0x0, 0x109, 0x203, 0x30a, 0x80c, 0x905, 0xa0f, 0xb06,
    0x406, 0x50f, 0x605, 0x70c, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x99c, 0x895, 0xb9f, 0xa96,
    0x596, 0x49f, 0x795, 0x69c, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0xa3c, 0xb35, 0x83f, 0x936,
    0x636, 0x73f, 0x435, 0x53c, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0xbac, 0xaa5, 0x9af, 0x8a6,
    0x7a6, 0x6af, 0x5a5, 0x4ac, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc, 0x1c5, 0x2cf, 0x3c6,
    0xcc6, 0xdcf, 0xec5, 0xfcc, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0x15c, 0x55, 0x35f, 0x256,
    0xd56, 0xc5f, 0xf55, 0xe5c, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0x2fc, 0x3f5, 0xff, 0x1f6,
    0xef6, 0xfff, 0xcf5, 0xdfc, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0x36c, 0x265, 0x16f, 0x66,
    0xf66, 0xe6f, 0xd65, 0xc6c, 0x76a, 0x663, 0x569, 0x460,
    0x460, 0x569, 0x663, 0x76a, 0xc6c, 0xd65, 0xe6f, 0xf66,
    0x66, 0x16f, 0x265, 0x36c, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x1f6, 0xff, 0x3f5, 0x2fc, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0xe5c, 0xf55, 0xc5f, 0xd56,
    0x256, 0x35f, 0x55, 0x15c, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0xfcc, 0xec5, 0xdcf, 0xcc6,
    0x3c6, 0x2cf, 0x1c5, 0xcc, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x4ac, 0x5a5, 0x6af, 0x7a6,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x53c, 0x435, 0x73f, 0x636,
    0x936, 0x83f, 0xb35, 0xa3c, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0x69c, 0x795, 0x49f, 0x596,
    0xa96, 0xb9f, 0x895, 0x99c, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0x70c, 0x605, 0x50f, 0x406,
    0xb06, 0xa0f, 0x905, 0x80c, 0x30a, 0x203, 0x109, 0x0,
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

// used for the mesh shader
groupshared float[MESHLET_VOLUME_PLUS_TWO] cached_density;
groupshared bool[MESHLET_VOLUME_PLUS_TWO] cached_is_set;
groupshared uint[MESHLET_VOLUME_PLUS_ONE] lookup;
groupshared Atomic<uint> vertex_count;
groupshared Atomic<uint> triangle_count;

struct DummyPayload {
    uint8_t[64] visible_meshlets;
    int3 offset_from_task_shader;
}

// used for the task shader
groupshared DummyPayload payload;
groupshared Atomic<uint> visible_meshlets;

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("task")]
[numthreads(64, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uniform int3 segment_offset) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    // chunk offset is from the groups of the task shader
    int3 chunk_offset = (int3)gid + segment_offset * 4;

    if (gtid.x == 0) {
        payload = DummyPayload({});
        payload.offset_from_task_shader = chunk_offset;
        visible_meshlets = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    int meshlet = gtid.x;

    uint3 meshlet_gid = index_to_offset(meshlet, 4);
    float3 meshlet_center = (float3)meshlet_gid * MESHLET_SIZE + (float3)CHUNK_SIZE * chunk_offset + (float3)(MESHLET_SIZE / 2); 

    bool meshlet_visible = true;   

    // frustum culling
    let camera_frustum_planes = uniforms.camera_frustum_planes;
    for (int f = 0; f < 6; f++) {
        float3 plane_normal = camera_frustum_planes[f].xyz;
        float plane_distance = camera_frustum_planes[f].w;

        if ((dot(plane_normal, meshlet_center) + plane_distance) < (-0.2 * 2.0)) {
            meshlet_visible = false;
            break;
        }
    }

    if (meshlet_visible) {
        uint index = visible_meshlets.increment(MemoryOrder.Relaxed);
        payload.visible_meshlets[index] = (uint8_t)meshlet;
    }

    GroupMemoryBarrierWithGroupSync();

    if (gtid.x == 0) {
        DispatchMesh(visible_meshlets.load(MemoryOrder.Relaxed), 1, 1, payload);
    }
}

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(6, 6, 6)]
void mesh_main(out vertices VertexOutput vertices[128], out indices uint3 triangles[256], uint3 gtid : SV_GroupThreadID, uint3 packed_gid : SV_GroupID, payload DummyPayload payload) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    
    // initialize group shared memory
    if (all(gtid == 0)) {
        lookup = {};
        vertex_count = 0;
        triangle_count = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    int3 chunk_offset = payload.offset_from_task_shader;
    
    // group ID is "packed" from the task shader
    // we need to unpack and do remapping to the original 3D meshlet group ID
    uint3 gid = index_to_offset(payload.visible_meshlets[packed_gid.x], 4);

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
    uint8_t bitmask = 0;
    if (all(gtid < MESHLET_SIZE_PLUS_ONE)) {
        for (int neighbour = 0; neighbour < 8; neighbour++) {
            uint3 neighbour_offset = index_to_offset(neighbour, 2);

            if (cached_is_set[offset_to_index(cell + neighbour_offset, MESHLET_SIZE_PLUS_TWO)]) {
                bitmask |= 1 << neighbour;
            }
        }
    }

    if (all(gtid < MESHLET_SIZE_PLUS_ONE) && bitmask != 0 && bitmask != 0xFF) {
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
            vertices[vertex_index].meshlet_colour = hash33(packed_gid);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // create triangles (up to 3 quads) per thread 
    if (all(gtid < MESHLET_SIZE_PLUS_ONE) && bitmask != 0 && bitmask != 0xFF) {
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

                /*
                float3 normal = 0;
                
                if (cell_set) {
                    normal[axis] = 1;
                } else {
                    normal[axis] = -1;
                }

                float3 view = normalize(uniforms.position.xyz - (cell + 0.5 + gid * MESHLET_SIZE + (int3)CHUNK_SIZE * chunk_offset));
                
                // TODO: perhaps we can do some backface culling here???
                if (dot(view, normal) < 0.0) {
                    continue;
                }
                */

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

    float3 reflection = sample_skybox_and_clouds(world, reflect(view, normal), clouds_sampler, skybox_sampler);

    if (uniform_buffer.debug_type % 2 == 0) {
        return float4(in_vert.meshlet_colour.xyz, 1);
    } else {
        return float4(reflection, 1);
    }
}