#language slang 2026
#include "utils/pbr.slang"

#include "utils/bindless.slang"

struct VertexOutput {
    float3 world_pos;
    float3 meshlet_colour;
    float4 sv_position : SV_Position;
};

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
static const float4 positions[3] = {
  float4( 0.0, -1.0, 0.0, 1.0),
  float4(-1.0, 1.0, 0.0, 1.0),
  float4( 1.0, 1.0, 0.0, 1.0)
};

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[66], out indices uint3 triangles[64], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, payload DummyPayload muh_payload) {
    // get buffers and preamble stuff (I <3 bindless)
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let tess_buffer = storage_buffers_readonly[3];
    float3[3] verts = muh_payload.verts;

    /*
    // passthrough!!!
    // calculate world space and clip space position of tesselated vertex (per thread!)
    float4 world_position = float4(verts[gtid.x], 1);
    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;
    vertices[gtid.x].world_pos = world_position.xyz;
    vertices[gtid.x].sv_position = t;
    vertices[gtid.x].meshlet_colour = hash33(gtid);
    triangles[0] = uint3(0, 1, 2);
    SetMeshOutputCounts(3, 1);
    */

    
    /*
    // apply first order tesselation (emitted by task shader)
    let fixed_header = tess_buffer.Load<Header>(4*4*muh_payload.fixed_task_lod);
    uint3 fixed_tri = tess_buffer.Load<uint3>(fixed_header.triangle_array_byte_offset + id.z * 4 * 3);
    for (int i = 0; i < 3; i++) {
        float3 weights_fo = tess_buffer.Load<float3>(fixed_header.vertex_array_byte_offset + 4 * 3 * fixed_tri[i]);
        verts[i] = starting_verts[0] * weights_fo.x + starting_verts[1] * weights_fo.y + starting_verts[2] * weights_fo.z;
    } 
    */   
      
    // fetch header for LOD from CPU generated tesselation buffer
    let header = tess_buffer.Load<Header>(4*4*muh_payload.lod);    
    
    for (int vertex_index = 0; vertex_index < header.vertex_count; vertex_index++) {
        // calculate world space and clip space position of tesselated vertex (per thread!)
        float3 weights = tess_buffer.Load<float3>(header.vertex_array_byte_offset + vertex_index * 4 * 3);
        float3 weighted_sum = verts[0] * weights.x + verts[1] * weights.y + verts[2] * weights.z;
        float4 world_position = float4(weighted_sum, 1);

        // some displacement perhaps
        //world_position.xyz += (hash33(world_position.xyz) - 0.5) * 0.2;

        float4 t = mul(world_position, uniforms.view_projection_matrix);
        t.y = -t.y;
        t.x = -t.x;

        vertices[vertex_index].world_pos = world_position.xyz;
        vertices[vertex_index].sv_position = t;
        vertices[vertex_index].meshlet_colour = muh_payload.lod / 3.0;
    }
    
    for (int triangle_index = 0; triangle_index < header.triangle_count; triangle_index++) {
        uint3 tri = tess_buffer.Load<uint3>(header.triangle_array_byte_offset + triangle_index * 4 * 3);
        triangles[triangle_index] = tri;
    }
    
    SetMeshOutputCounts(header.vertex_count, header.triangle_count);
    

    /*
    float4 world_position = positions[gtid.x];
    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    vertices[gtid.x].world_pos = world_position.xyz;
    vertices[gtid.x].sv_position = t;
    vertices[gtid.x].meshlet_colour = hash33(gtid);

    triangles[0] = uint3(0, 1, 2);

    SetMeshOutputCounts(3, 1);
    */
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

    
    /*
    float3 diffuse = 0.5;
    float roughness = 0.8;
    float metallic = 0.2;
    float ao = 1.0;
    float shadow = 1.0; 

    float3 ray_pos = uniform_buffer.position.xyz;

    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
    CameraData camera_data = CameraData(view);

    // approximate sun color based on height
    float3 sun = uniform_buffer.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);
    SkyData sky_data = SkyData(skybox_sampler, clouds_sampler);

    float3 colour = lighting(pbr_surface_data, camera_data, sun_data, sky_data);
    return float4(colour, 1.0);
    */
    
    // float3 reflection = sample_skybox_and_clouds(world, refract(view, normal, 0.95), clouds_sampler, skybox_sampler);
    // return float4(reflection, 1);
    return float4(in_vert.meshlet_colour, 1);
}