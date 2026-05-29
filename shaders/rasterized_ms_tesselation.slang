#language slang 2026
#include "utils/pbr.slang"
#include "utils/mesh_shader_commons.slang"
#include "utils/bindless.slang"

struct DummyPayload {
    float3[3] verts;
    uint lod;
}


// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("task")]
[numthreads(1, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    int dispatch_mesh_size = 1;

    // get buffers and preamble stuff (I <3 bindless)
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let vertex_buffer = storage_buffers_readonly[1];
    let index_buffer = storage_buffers_readonly[2];
    let tess_buffer = storage_buffers_readonly[3];
    
    //float3 instance_offset = float3(gid.y * 10, 0, 0);
    float3 instance_offset = float3(0, id.y * 5, id.z * 5);
    
    // using triangle index, load the triangle's vertices into a array
    uint3 loaded_triangle = index_buffer.Load3(gid.x * 4 * 3);
    float3[3] verts = {};    
    for (int i = 0; i < 3; i++) {
        float3 loaded_vertex = vertex_buffer.Load<float3>(loaded_triangle[i] * 4 * 3);
        verts[i] = loaded_vertex + instance_offset;
    }    

    DummyPayload payload = DummyPayload(verts);
    float3 center = (verts[0] + verts[1] + verts[2]) / 3.0;
    float3 view = normalize(center - uniforms.position.xyz);

    // backface culling
    // https://stackoverflow.com/questions/19350792/calculate-normal-of-a-single-triangle-in-3d-space
    float3 a = verts[2] - verts[0];
    float3 b = verts[1] - verts[0];
    float3 normal = normalize(cross(a,b));
    if (dot(normal, view) < 0.0) {
        dispatch_mesh_size = 0;
    }
    
    // frustum culling
    let camera_frustum_planes = uniforms.camera_frustum_planes;
    for (int f = 0; f < 6; f++) {
        float3 plane_normal = camera_frustum_planes[f].xyz;
        float plane_distance = camera_frustum_planes[f].w;

        if ((dot(plane_normal, center) + plane_distance) < (-0.2 * 2.0)) {
            dispatch_mesh_size = 0;
            break;
        }
    }

    // calculate LOD level based on distance to center of triangle
    float3 aabb_min = min3(verts[0], verts[1], verts[2]);
    float3 aabb_max = max3(verts[0], verts[1], verts[2]);


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
    float screen_heuristic = length(size_ndc) * 25.0;
    int lod = clamp(floor(screen_heuristic), 0, 3);
    payload.lod = lod;

    DispatchMesh(dispatch_mesh_size, 1, 1, payload);
}

struct Header {
    uint32_t vertex_count;
    uint32_t triangle_count;
    uint32_t vertex_array_byte_offset;
    uint32_t triangle_array_byte_offset;
}

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[66], out indices uint3 triangles[64], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, payload DummyPayload muh_payload) {
    // get buffers and preamble stuff (I <3 bindless)
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let tess_buffer = storage_buffers_readonly[3];
    float3[3] verts = muh_payload.verts;
      
    // fetch header for LOD from CPU generated tesselation buffer
    let header = tess_buffer.Load<Header>(4*4*muh_payload.lod);    
    
    for (int vertex_index = 0; vertex_index < header.vertex_count; vertex_index++) {
        float3 weights = tess_buffer.Load<float3>(header.vertex_array_byte_offset + vertex_index * 4 * 3);
        float3 weighted_sum = verts[0] * weights.x + verts[1] * weights.y + verts[2] * weights.z;
        vertices[vertex_index] = transform_vertex(weighted_sum, muh_payload.lod / 3.0);
    }
    
    for (int triangle_index = 0; triangle_index < header.triangle_count; triangle_index++) {
        uint3 tri = tess_buffer.Load<uint3>(header.triangle_array_byte_offset + triangle_index * 4 * 3);
        triangles[triangle_index] = tri;
    }
    
    SetMeshOutputCounts(header.vertex_count, header.triangle_count);
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    return shared_frag_shading(in_vert);
}