#language slang 2026
#include "utils/noises.slang"
#include "utils/bindless.slang"
#include "utils/mesh_shader_commons.slang"


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
    let index_buffer = storage_buffers_readonly[STORAGE_BUFFER_INDEX_BUFFER_IDX];

    // load triangle from index buffer
    // this triangle indexes into the global vertex buffer
    uint3 triangle = index_buffer.Load<uint3>(id.x * 4 * 3);
    
    // "local" in terms of the triangle (so in range in [0..2])
    for (int local_vertex_index = 0; local_vertex_index < 3; local_vertex_index++) {
        // using thread ID, load global vertex
        // yes this will duplicate a lot of the vertex transformation stuff
        float3 loaded_vertex = vertex_buffer.Load<float3>(triangle[local_vertex_index] * 4 * 3);

        // the threads inside the same group cooperate to write the mini vertex buffer
        vertices[local_vertex_index + gtid.x * 3] = transform_vertex(loaded_vertex, hash33(gid));
    }
    
    triangles[gtid.x] = uint3(0, 1, 2) + gtid.x * 3;
    
    let generated_triangles = loaded_triangles_in_group.load(MemoryOrder.Relaxed);
    
    SetMeshOutputCounts(generated_triangles*3, generated_triangles);
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    return shared_frag_shading(in_vert);
}