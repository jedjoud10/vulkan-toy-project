#language slang 2026
#include "utils/other.slang"
#include "utils/pbr.slang"
#include "utils/bindless.slang"
#include "utils/mesh_shader_commons.slang"

struct TaskShaderPayload {
    uint2 patch_offset;
}

[shader("task")]
[numthreads(1, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    TaskShaderPayload payload = TaskShaderPayload(gid.xy);
    DispatchMesh(8, 8, 1, payload);
}

// https://gpuopen.com/learn/mesh_shaders/mesh_shaders-procedural_grass_rendering/
float3 bezier(float3 p0, float3 p1, float3 p2, float t) {
    float3 a = lerp(p0, p1, t);
    float3 b = lerp(p1, p2, t);
    return lerp(a, b, t);
}

const static uint NUM_QUADS_PER_BLADE = 3;
const static uint NUM_VERTICES_PER_BLADE = 8;
const static uint NUM_BLADES_PER_MS_GROUP = 8;
const static float PATCH_SIZE_WORLD_SIZE = 2.0;
const static float WIDTH = 0.12;

// x coord will be multiplided by side to account for blade symmetry
const static float3 p0 = float3(1.0, 0.0, 0.0);
const static float3 p1 = float3(0.8, 0.5, 0.0);
const static float3 p2 = float3(0.6, 1.0, -0.2);


[shader("mesh")]
[outputtopology("triangle")]
[numthreads(8, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[64], out indices uint3 triangles[64], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 packed_gid : SV_GroupID, payload TaskShaderPayload payload) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    
    // generate vertices
    for (int i = 0; i < NUM_VERTICES_PER_BLADE; i++) {
        float factor = (float)(i / 2) / (NUM_VERTICES_PER_BLADE / 2);

        float side = ((i % 2) * 2.0 - 1.0) * WIDTH;
        float3 local_position = bezier(p0, p1, p2, factor);
        local_position.x *= side;

        // unique for each blade in the world
        float2 id_hash_input = id.x * 1.44325 + id.y * 143.44325 + payload.patch_offset * 5.2323;

        // apply x-z rotation
        float theta = hash12(id_hash_input) * 3.1415 * 2.0;
        local_position.xz = mul(float2x2(cos(theta), -sin(theta), sin(theta), cos(theta)), local_position.xz);
        float3 world_position = local_position;
        
        // random offset per grass blade
        world_position.xz += hash22(id_hash_input) * PATCH_SIZE_WORLD_SIZE;

        // apply patch ofset
        world_position.xz += payload.patch_offset * PATCH_SIZE_WORLD_SIZE;

        VertexOutput vert = transform_vertex(world_position, 0);
        vert.user = factor;
        vertices[gtid.x * NUM_VERTICES_PER_BLADE + i] = vert;
    }
    
    // generate triangle quads
    for (int quad_index = 0; quad_index < NUM_QUADS_PER_BLADE; quad_index++) {
        triangles[gtid.x * NUM_QUADS_PER_BLADE + quad_index*2] = uint3(0,1,2) + quad_index*2 + gtid.x * NUM_VERTICES_PER_BLADE;
        triangles[gtid.x * NUM_QUADS_PER_BLADE + quad_index*2+1] = uint3(2,1,3) + quad_index*2 + gtid.x * NUM_VERTICES_PER_BLADE;
    }

    SetMeshOutputCounts(NUM_VERTICES_PER_BLADE*NUM_BLADES_PER_MS_GROUP, NUM_BLADES_PER_MS_GROUP*NUM_QUADS_PER_BLADE*2);
}

float3 parse_srgb_hex_colour_to_linear(uint hex) {
    uint3 zero_to_byte = uint3(hex >> 16, (hex >> 8) & 0xFF, hex & 0xFF);
    return pow((float3)zero_to_byte / 255.0, 2.2);
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    let factor = in_vert.user.x;

    float3 bottom_grass_colour = parse_srgb_hex_colour_to_linear(0x3e6d27);
    float3 top_grass_colour = parse_srgb_hex_colour_to_linear(0x589b36);


    return float4(lerp(bottom_grass_colour, top_grass_colour, factor), 1.0);
}

/*
// each mesh shader group will generate 8 blades of grass, each having 8 vertices and 3 quads
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[64], out indices uint3 triangles[128], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 packed_gid : SV_GroupID, payload TaskShaderPayload payload) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    let i = (int)gtid.x;
    let blade_grass_index = i / NUM_VERTICES_PER_BLADE;
    
    float factor = (float)(i / 2) / (NUM_VERTICES_PER_BLADE / 2);

    float side = ((i % 2) * 2.0 - 1.0) * WIDTH;
    float3 local_position = bezier(p0, p1, p2, factor);
    local_position.x *= side;

    // unique for each blade in the world
    float2 id_hash_input = blade_grass_index * 143.44325 + payload.patch_offset * 5.2323;

    // apply x-z rotation
    float theta = hash12(id_hash_input) * 3.1415 * 2.0;
    local_position.xz = mul(float2x2(cos(theta), -sin(theta), sin(theta), cos(theta)), local_position.xz);
    float3 world_position = local_position;
    
    // random offset per grass blade
    //world_position.xz += hash22(id_hash_input) * PATCH_SIZE_WORLD_SIZE;

    // apply patch offset
    //world_position.xz += payload.patch_offset * PATCH_SIZE_WORLD_SIZE;

    VertexOutput vert = transform_vertex(world_position, 0);
    vert.user = factor;
    vertices[i] = vert;

    if (gtid.x == 0) {
        triangles[0*2] = uint3(0,1,2) + 0*2;
        triangles[0*2+1] = uint3(2,1,3) + 0*2;
    }

    if (i < NUM_QUADS_PER_BLADE * NUM_BLADES_PER_MESH_DISPATCH_GROUP) {
        triangles[i*2] = uint3(0,1,2) + i*2;
        triangles[i*2+1] = uint3(2,1,3) + i*2;
    }

    SetMeshOutputCounts(NUM_BLADES_PER_MESH_DISPATCH_GROUP * NUM_VERTICES_PER_BLADE, NUM_QUADS_PER_BLADE * NUM_BLADES_PER_MESH_DISPATCH_GROUP * 2);
}
*/