#language slang 2026

#include "utils/noises.slang"
#include "utils/sky.slang"
#include "utils/bindless.slang"

const static uint MESHLET_SIZE = 4;

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


float density(float3 p) {
    float value = 0;

    float scale = 0.02;
    float amplitude = 20.0;
    float3 offset = 0.0;

    for (int i = 0; i < 7; i++) {
        value += abs(snoise(scale * p + offset)) * amplitude;

        scale *= 1.7;
        amplitude *= 0.4;
        offset += snoise((float)i * 0.5 + p * 0.02) * 0.6;
    }

    /*
    if (length(p) < 30) {
        value = 20 + snoise(-p * 0.2 + 4.12312);

        if (length(p) > 18 + snoise(p * 0.2)) {
            value = -20;
        }
    }
    */

    //value += snoise(p * 0.01 * float3(1, 0.2, 1) + snoise(p * 0.03) * 0.2) * 50; 
    
    /*
    if (snoise(p.xz * 0.005) > 0) {
        value += snoise(p * 0.05 + snoise(p * 0.01) * 0.2) * 30; 
    } else {
        value += -cellular(p * 0.01).y * 200; 
    }
    */
    
    //value += snoise(p * 0.1);
    /*
    if (hash12(floor(p.xz * 0.05)) > 0.8) {
        value = snoise(p * 0.1) * 15;
    }
    */
    value += p.y - 10.0;
    return value;
}

float3 normal_from_density(float3 p) {
    float base = density(p);

    float eps = 0.001;
    float x = density(p + float3(eps, 0, 0));
    float y = density(p + float3(0, eps, 0));
    float z = density(p + float3(0, 0, eps));
    
    return -normalize(float3(x,y,z) - base);
}


[shader("compute")]
[numthreads(4, 4, 4)]
void calculate_density(uint3 id: SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uniform int3 chunk_offset) {
    let voxel_image = storage_images_3d_float[VOXEL_STORAGE_IMAGE_IDX];
    let uniforms = load_per_frame_uniforms();
    
    float3 p = id + (float3)chunk_offset * 64;

    float density = density(p);

    //float value = snoise(p * 0.5);
    voxel_image.Store(id, density);
}

groupshared Atomic<uint32_t> positive;
groupshared Atomic<uint32_t> negative;

[shader("compute")]
[numthreads(6, 6, 6)]
void voxelize_surface(uint3 id: SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uniform int3 chunk_offset) {
    let index_counter = storage_buffers[STORAGE_BUFFER_INDEX_COUNTER_IDX];
    let vertex_counter = storage_buffers[STORAGE_BUFFER_VERTEX_COUNTER_IDX];
    let vertex_buffer = storage_buffers[STORAGE_BUFFER_VERTEX_BUFFER_IDX];
    let index_buffer = storage_buffers[STORAGE_BUFFER_INDEX_BUFFER_IDX];
    let voxel_image = readonly_storage_images_3d_float[VOXEL_STORAGE_IMAGE_IDX];

    // initialize group shared memory
    if (all(gtid == 0)) {
        lookup = {};
        positive = false;
        negative = false;
    }
    
    // port of https://github.com/jedjoud10/vulkan-toy-project/blob/main/src/voxel/meshes.rs
    // generate vertices
    uint3 cell = gtid;

    // cache density function and booleans
    float cell_density = voxel_image.Load(gid * 4 + gtid);
    bool cell_set = cell_density < 0;
    int groupshared_memory_index = offset_to_index(cell, MESHLET_SIZE_PLUS_TWO);
    cached_is_set[groupshared_memory_index] = cell_set;
    cached_density[groupshared_memory_index] = cell_density;
    positive.or(cell_set, MemoryOrder.Relaxed);
    negative.or(!cell_set, MemoryOrder.Relaxed);

    GroupMemoryBarrierWithGroupSync();

    // the entire group invocation should get discarded when we know there won't be a surface generated
    bool positive = (bool)positive.load(MemoryOrder.Relaxed);
    bool negative = (bool)negative.load(MemoryOrder.Relaxed);
    bool surf = positive && negative;
    if (!surf) {
        return;
    }
    
    // not diverging because the *entire* group would have gone discarded. there are no diverging paths within a group
    GroupMemoryBarrierWithGroupSync();

    // create single vertex per thread
    if (all(gtid < MESHLET_SIZE_PLUS_ONE)) {
        float3 inside_cell_offset_avg = 0;
        int count = 0;


        for (int edge = 0; edge < 12; edge++) {
            uint3 p1_offsetted = EDGE_POSITIONS_0[edge];
            uint3 p2_offsetted = EDGE_POSITIONS_1[edge];

            uint p1_offsetted_index = offset_to_index(cell + p1_offsetted, MESHLET_SIZE_PLUS_TWO);
            uint p2_offsetted_index = offset_to_index(cell + p2_offsetted, MESHLET_SIZE_PLUS_TWO);
            
            float d1 = cached_density[p1_offsetted_index];
            float d2 = cached_density[p2_offsetted_index];
            
            if (d1 < 0 ^ d2 < 0) {
                float factor = unlerp(0, d1, d2);
                inside_cell_offset_avg += lerp(p1_offsetted, p2_offsetted, factor);
                count++;
            }
        }

        uint vertex_index;
        vertex_counter.InterlockedAdd(0, 1, vertex_index); 
        lookup[offset_to_index(cell, MESHLET_SIZE_PLUS_ONE)] = vertex_index;

        float3 cell_offset = cell + (inside_cell_offset_avg / count);
        float3 vertex_position = (cell_offset + gid * MESHLET_SIZE + chunk_offset * 64);

        
        float3 normal = normalize(normal_from_density(vertex_position));
        
        vertex_buffer.Store<float3>( vertex_index * 4 * 3, vertex_position);
        //uint half3_stride = 2 * 3;
        //vertex_buffer.Store<half3>(2 * vertex_index * half3_stride, (half3)vertex_position);
        //vertex_buffer.Store<half3>(2 * vertex_index * half3_stride + half3_stride, (half3)normal);
        
    }

    GroupMemoryBarrierWithGroupSync();

    // create triangles (up to 3 quads) per thread 
    if (all(gtid < MESHLET_SIZE_PLUS_ONE)) {
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
            
                // TODO: since we know which vertices are which at this point, we can change the triangle ordering to preserve topoly better
                // this should remedy the "spike" problem that happens when we get a quad where two opposing corner vertices are significantly higher/lower than the other two
                // I am unsure of what heuristic to use to detect that though
                uint index_index;
                index_counter.InterlockedAdd(0, 6, index_index);
                uint triangle_index = index_index / 3;
                if (dir) {
                    index_buffer.Store3(3 * 4 * triangle_index, uint3(quad_vertex_indices.x, quad_vertex_indices.y, quad_vertex_indices.z));
                    index_buffer.Store3(3 * 4 * (triangle_index + 1), uint3(quad_vertex_indices.z, quad_vertex_indices.y, quad_vertex_indices.w));
                } else {
                    index_buffer.Store3(3 * 4 * triangle_index, uint3(quad_vertex_indices.y, quad_vertex_indices.x, quad_vertex_indices.z));
                    index_buffer.Store3(3 * 4 * (triangle_index + 1), uint3(quad_vertex_indices.y, quad_vertex_indices.z, quad_vertex_indices.w));
                }
            }
        }
    }
}