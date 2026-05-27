#language slang 2026
#include "utils/pbr.slang"
#include "utils/uniform_thing.slang"
#include "utils/bindless.slang"

struct VertexOutput {
    float3 world_pos;
    float3 meshlet_colour;
    float4 sv_position : SV_Position;
};

groupshared Atomic<uint> loaded_triangles_in_group;

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[96], out indices uint3 triangles[32], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uniform uint triangle_count_mesh) {
    if (gtid.x == 0) {
        loaded_triangles_in_group = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    if (id.x < triangle_count_mesh) {
        loaded_triangles_in_group.add(1, MemoryOrder.Relaxed);
    } else {
        return;
    }

    GroupMemoryBarrierWithGroupSync();

    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let vertex_buffer = storage_buffers_readonly[STORAGE_BUFFER_VERTEX_BUFFER_IDX];
    let index_buffer = storage_buffers_readonly[STORAGE_BUFFER_INDEX_BUFFER];

    // load triangle from index buffer
    // this triangle indexes into the global vertex buffer
    uint3 triangle = index_buffer.Load<uint3>(id.x * 4 * 3);
    
    // "local" in terms of the triangle (so in range in [0..2])
    for (int local_vertex_index = 0; local_vertex_index < 3; local_vertex_index++) {
        
        // using thread ID, load global vertex
        // yes this will duplicate a lot of the vertex transformation stuff
        float3 loaded_vertex = vertex_buffer.Load<float3>(triangle[local_vertex_index] * 4 * 3);

        float4 world_position = float4(loaded_vertex, 1);

        float4 t = mul(world_position, uniforms.view_projection_matrix);
        t.y = -t.y;
        t.x = -t.x;

        // the threads inside the same group cooperate to write the mini vertex buffer
        vertices[local_vertex_index + gtid.x * 3].world_pos = world_position.xyz;
        vertices[local_vertex_index + gtid.x * 3].sv_position = t;
        vertices[local_vertex_index + gtid.x * 3].meshlet_colour = hash33(gid);

    }
    
    triangles[gtid.x] = uint3(0, 1, 2) + gtid.x * 3;
    
    let generated_triangles = loaded_triangles_in_group.load(MemoryOrder.Relaxed);
    
    SetMeshOutputCounts(generated_triangles*3, generated_triangles);
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

    float3 refraction = sample_skybox_and_clouds(world, refract(view, normal, 0.95), clouds_sampler, skybox_sampler);
    return float4(refraction, 1);
}