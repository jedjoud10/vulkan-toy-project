#language slang 2026
#include "utils/other.slang"
#include "utils/pbr.slang"
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
    
    float warp_scale = 0.2;
    float3 warp = float3(snoise(p*warp_scale + -2.213 + uniforms.time * 1.5), snoise(p*warp_scale + 4.12 -  uniforms.time * 1.01), snoise(p*warp_scale + 3.123 +  uniforms.time * 1.04));

    float density = p.y - (2 + snoise(p * 0.02 + warp * 0.0) * 32);

    float sphere = length(p - uniforms.position.xyz) - 10;
    density = max(density, -sphere);
    return density;
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


const static uint MESHLET_SIZE = 4;
const static uint CHUNK_SIZE = MESHLET_SIZE * 4;

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

// TODO: possible fast path using wave intrinsics instead?
struct DummyPayload {
    uint8_t[64] visible_meshlets;
    int3 offset_from_task_shader;
    int3 task_shader_gid;
    int scale;
}

// used for the task shader
groupshared DummyPayload payload;
groupshared Atomic<uint> visible_meshlets;

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("task")]
[numthreads(64, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uniform int3 chunk_offset_cs) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    // chunk offset is from the groups of the task shader
    int3 meshlet_offset = (int3)gid + chunk_offset_cs * 4;
    float3 chunk_offset_ws = (float3)CHUNK_SIZE * meshlet_offset; 
    float3 chunk_center = (float3)CHUNK_SIZE * meshlet_offset + (float3)CHUNK_SIZE * 0.5; 

    /*
    // calculate LOD level based on distance to center of triangle
    float3 aabb_min = (float3)chunk_offset * CHUNK_SIZE;
    float3 aabb_max = (float3)chunk_offset * CHUNK_SIZE + (float3)CHUNK_SIZE;

    float3 aabb_corners[8];
    aabb_corners[0] = float3(aabb_min.x, aabb_min.y, aabb_min.z);
    aabb_corners[1] = float3(aabb_max.x, aabb_min.y, aabb_min.z);
    aabb_corners[2] = float3(aabb_min.x, aabb_max.y, aabb_min.z);
    aabb_corners[3] = float3(aabb_max.x, aabb_max.y, aabb_min.z);
    aabb_corners[4] = float3(aabb_min.x, aabb_min.y, aabb_max.z);
    aabb_corners[5] = float3(aabb_max.x, aabb_min.y, aabb_max.z);
    aabb_corners[6] = float3(aabb_min.x, aabb_max.y, aabb_max.z);
    aabb_corners[7] = float3(aabb_max.x, aabb_max.y, aabb_max.z);

    float2 ndc_min = float2(1e9, 1e9);
    float2 ndc_max = float2(-1e9, -1e9);
    for (int i = 0; i < 8; i++) {
        float4 c = mul(float4(aabb_corners[i], 1.0), uniforms.view_projection_matrix);
        float2 ndc = c.xy / c.w;
        ndc_min = min(ndc_min, ndc);
        ndc_max = max(ndc_max, ndc);
    }

    float2 size_ndc = ndc_max - ndc_min;
    float screen_heuristic = 40.0 - length(size_ndc) * 10.0;


    // this must be the same for the entire task shader group!
    int lod = floor(clamp(log2(screen_heuristic), 0, 2));
    */


    // this must be the same for the entire task shader group!
    int lod = floor(clamp(distance(chunk_center, uniforms.position.xyz) / 100, 0, 2));
    int scale = 1 << lod;

    if (gtid.x == 0) {
        payload = DummyPayload({});
        payload.scale = scale;
        payload.offset_from_task_shader = meshlet_offset;
        payload.task_shader_gid = gid;
        visible_meshlets = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    int meshlet = gtid.x;

    uint3 meshlet_gid = index_to_offset(meshlet, 4);

    float sphere_radius = CHUNK_SIZE;

    // toggle debug frustum culling
    if (uniforms.toggles_bitmask & 1 == 1) {
        sphere_radius = 0.1;
    }

    // if LOD=0, we want all 4x4x4 meshlets to execute
    // if LOD=1, we only want 2x2x2 meshlets to execute, those at the even coordinates (since each are now twice as big)
    // if LOD=2, we only want 1 meshlet to execute and that one will represent the entire chunk that would have been spawned by one task shader group
    if (all((meshlet_gid % scale) == 0)) {
        float3 meshlet_center = (float3)meshlet_gid * MESHLET_SIZE + chunk_offset_ws + (float3)(MESHLET_SIZE / 2) * scale; 

        bool meshlet_visible = true;   

        // frustum culling
        let camera_frustum_planes = uniforms.camera_frustum_planes;
        for (int f = 0; f < 6; f++) {
            float3 plane_normal = camera_frustum_planes[f].xyz;
            float plane_distance = camera_frustum_planes[f].w;

            if ((dot(plane_normal, meshlet_center) + plane_distance) < (-sphere_radius * 2.0)) {
                meshlet_visible = false;
                break;
            }
        }

        if (meshlet_visible) {
            uint index = visible_meshlets.increment(MemoryOrder.Relaxed);
            payload.visible_meshlets[index] = (uint8_t)meshlet;
        }
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

    int3 chunk_offset = payload.offset_from_task_shader;
    
    // group ID is "packed" from the task shader
    // we need to unpack and do remapping to the original 3D meshlet group ID
    uint3 gid = index_to_offset(payload.visible_meshlets[packed_gid.x], 4);

    // port of https://github.com/jedjoud10/vulkan-toy-project/blob/main/src/voxel/meshes.rs
    // generate vertices
    uint3 cell = gtid;

    // cache density function and booleans
    float cell_density = density(cell * payload.scale, gid, chunk_offset);
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

        let vertex_index = vertex_count.increment(MemoryOrder.Relaxed);
        lookup[offset_to_index(cell, MESHLET_SIZE_PLUS_ONE)] = vertex_index;

        float3 cell_offset = cell + (inside_cell_offset_avg / count);
        cell_offset = cell + 0.5;

        float3 loaded_vertex = cell_offset * payload.scale /* + floor(uniforms.position.xyz) */ + gid * MESHLET_SIZE + (int3)CHUNK_SIZE * chunk_offset;

        float4 world_position = float4(loaded_vertex, 1);
        float4 t = mul(world_position, uniforms.view_projection_matrix);
        t.y = -t.y;
        t.x = -t.x;
        vertices[vertex_index].world_pos = world_position.xyz;
        vertices[vertex_index].sv_position = t;

        float3 colour = 0;
        if (is_bit_set(uniforms.toggles_bitmask, 2)) {
            colour = hash33(payload.task_shader_gid);
        } else {
            colour = hash33((int3)gid + chunk_offset + payload.scale);
        }
        vertices[vertex_index].meshlet_colour = colour;
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
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let skybox_sampler = sampled_images_cube[SKYBOX_SAMPLER_IMAGE_IDX];
    let clouds_sampler = sampled_images[CLOUDS_SAMPLER_IMAGE_IDX];
    
    float3 world = in_vert.world_pos;
    
    //float3 normal = normalize(in_vert.normal + float3(snoise(world.xz * 0.124) * 0.01));
    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));
    float3 view = normalize(in_vert.world_pos - uniforms.position.xyz);

    if (is_bit_set(uniforms.toggles_bitmask, 1)) {
        return float4(in_vert.meshlet_colour, 1);
    }

    /*
    float3 reflection = sample_skybox_and_clouds(world, reflect(view, normal), clouds_sampler, skybox_sampler);

    if (uniform_buffer.debug_type % 2 == 0) {
        return float4(in_vert.meshlet_colour.xyz, 1);
    } else {
        return float4(reflection, 1);
    }
        */

    float3 diffuse = 0.5;
    float roughness = 0.8;
    float metallic = 0.2;
    float ao = 1.0;
    float shadow = 1.0; 

    float3 ray_pos = uniforms.position.xyz;

    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
    CameraData camera_data = CameraData(view);

    // approximate sun color based on height
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);
    SkyData sky_data = SkyData(skybox_sampler, clouds_sampler);

    float3 colour = lighting(pbr_surface_data, camera_data, sun_data, sky_data);
    return float4(colour, 1.0);
}