#language slang 2026
#include "utils/other.slang"
#include "utils/pbr.slang"
#include "utils/bindless.slang"
#include "utils/mesh_shader_commons.slang"
#include "utils/culling.slang"

struct TaskShaderPayload {
    uint2 task_shader_offset;
    float fractional_z_workgroup_invocations;
}

[shader("task")]
[numthreads(1, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    
    float2 center_2d = gid.xy * SUPER_PATCH_SIZE_WORLD_SIZE + SUPER_PATCH_SIZE_WORLD_SIZE * 0.5;
    float3 center = float3(center_2d.x, 0, center_2d.y);
    
    float fractional_z_workgroup_invocations = clamp(16 - distance(center, uniforms.position.xyz) / 8.0 + 1.0, 0, 16);
    uint z_workgroup_invocations = (uint)ceil(fractional_z_workgroup_invocations);
    //z_workgroup_invocations = clamp(z_workgroup_invocations, 1, 8);
    
    //float width_multiplier = (8 - z_workgroup_invocations) + 1;
    float width_multiplier = 1.0;
    
    
    if (!frustum_cull_sphere_is_visible(center, box_aabb_half_extent_to_sphere_radius(SUPER_PATCH_SIZE_WORLD_SIZE * 0.5))) {
        z_workgroup_invocations = 0;
    }
    
    // 8x8 patches
    // z value can be used to increase density
    TaskShaderPayload payload = TaskShaderPayload(gid.xy, fractional_z_workgroup_invocations);
    DispatchMesh(8, 8, z_workgroup_invocations, payload);
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

const static float SUPER_PATCH_SIZE_WORLD_SIZE = PATCH_SIZE_WORLD_SIZE * 8.0;
const static float PATCH_SIZE_WORLD_SIZE = 2.0;
const static float WIDTH = 0.1;

const static float WIDTH_RANDOMIZATION = 0.015;
const static float HEIGHT_RANDOMIZATION = 0.3;
const static float BEND_RANDOMIZATION = 0.45;


// x coord will be multiplided by side to account for blade symmetry
const static float3 p0 = float3(1.0, 0.0, 0.0);
const static float3 p1 = float3(0.5, 0.4, 0.0);
const static float3 p2 = float3(0.3, 0.8, -0.5);



[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[64], out indices uint3 triangles[64], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, payload TaskShaderPayload payload) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    let blade_index_in_thread_group = gtid.x / NUM_VERTICES_PER_BLADE;
    let vertex_index_for_blade = gtid.x % NUM_VERTICES_PER_BLADE;
    
    float2 id_hash_input = blade_index_in_thread_group * 4.2312 + payload.task_shader_offset * float2(5.2323, -663.32) + gid.z * -14.232 + gid.xy * float2(-12.123, 3.5645);
    float2 grass_blade_world_offset = hash22(id_hash_input) * PATCH_SIZE_WORLD_SIZE + payload.task_shader_offset * SUPER_PATCH_SIZE_WORLD_SIZE + gid.xy * PATCH_SIZE_WORLD_SIZE;
    float theta = hash12(id_hash_input) * 3.1415 * 2.0;
    float3 changes = (snoise(grass_blade_world_offset * 0.2) * 0.5 + 0.5) * float3(WIDTH_RANDOMIZATION, HEIGHT_RANDOMIZATION, BEND_RANDOMIZATION);

    // calculate extra multiplier based on fractional z workgroup invocations
    float lod_scale_multiplier = saturate(payload.fractional_z_workgroup_invocations - gid.z);

    // generate vertices
    float factor = (float)(vertex_index_for_blade / 2) / (NUM_VERTICES_PER_BLADE / 2);
    float factor_horizontal = (vertex_index_for_blade % 2);
    float2 uv = float2(factor_horizontal, factor);
    float side = (factor_horizontal * 2.0 - 1.0) * WIDTH;

    float3 p1_modified = p1 + float3(0, changes.y, 0);
    float3 p2_modified = p2 + float3(0, changes.y * 2.0, changes.z);
    float3 local_position = bezier(p0, p1_modified, p2_modified, factor);
    local_position.x *= side + changes.x;        
    local_position.xz = mul(float2x2(cos(theta), -sin(theta), sin(theta), cos(theta)), local_position.xz);
    float3 world_position = local_position * lod_scale_multiplier;
    
    // apply grass blade ofset
    world_position.xz += grass_blade_world_offset;

    // apply wind?
    world_position.xz += sin(uniforms.time * 3.0 + snoise(grass_blade_world_offset * 0.2) * 1.0) * factor * 0.1;

    VertexOutput vert = transform_vertex(world_position, 0);
    vert.user = float4(uv, id_hash_input);
    vertices[gtid.x] = vert;
    
    // generate triangle quads but on a separate "stride" than we did the vertices
    // we have to do some packing on our own....
    // each thread will write 1 quads for the corresponding blade but not necessarily *that* thread's blade
    if (gtid.x < NUM_QUADS_PER_BLADE * NUM_BLADES_PER_MS_GROUP) {
        let target_blade_index_in_thread_group = gtid.x / NUM_QUADS_PER_BLADE;
        let quad_index_for_blade = gtid.x % NUM_QUADS_PER_BLADE;


        triangles[target_blade_index_in_thread_group * 2 * NUM_QUADS_PER_BLADE + quad_index_for_blade*2] = uint3(0,1,2) + quad_index_for_blade*2 + target_blade_index_in_thread_group * NUM_VERTICES_PER_BLADE;
        triangles[target_blade_index_in_thread_group * 2 * NUM_QUADS_PER_BLADE + quad_index_for_blade*2+1] = uint3(2,1,3) + quad_index_for_blade*2 + target_blade_index_in_thread_group * NUM_VERTICES_PER_BLADE;
    }

    SetMeshOutputCounts(NUM_VERTICES_PER_BLADE*NUM_BLADES_PER_MS_GROUP, NUM_BLADES_PER_MS_GROUP*NUM_QUADS_PER_BLADE*2);
}

float3 parse_srgb_hex_colour_to_linear(uint hex) {
    uint3 zero_to_byte = uint3(hex >> 16, (hex >> 8) & 0xFF, hex & 0xFF);
    return pow((float3)zero_to_byte / 255.0, 2.2);
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    let factor = in_vert.user.y;

    float3 bottom_grass_colour = parse_srgb_hex_colour_to_linear(0x0a4920) * 0.3 * (0.4 + hash12(in_vert.user.zw) * 0.2);
    float3 top_grass_colour = parse_srgb_hex_colour_to_linear(0x589b36) * (0.8 + hash12(in_vert.user.zw) * 0.2);
    float3 subsurface_colour = parse_srgb_hex_colour_to_linear(0x91e069);
    float3 colour = lerp(bottom_grass_colour, top_grass_colour, factor);
    
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));
    float3 view = normalize(in_vert.world_pos - uniforms.position.xyz);


    let n_dot_l = saturate(abs(dot(uniforms.sun.xyz, normal)));
    let v_dot_l = saturate(dot(uniforms.sun.xyz, view));
    float shadow = 1.0 - clouds(uniforms.sun.xyz, in_vert.world_pos, sampled_images[CLOUDS_SAMPLER_IMAGE_IDX]).w;

    let combined = 1.0;
    let subsurface = pow(v_dot_l, 8) * 2.0 * subsurface_colour * 0.0;
    
    return float4(colour * combined + subsurface * .0, 1.0);
}