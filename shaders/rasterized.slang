#language slang 2026
#include <pbr.slang>
#include <uniform_thing.slang>
#include <ray_stuff_other.slang>
#include <bindless.slang>

struct VertexOutput {
    float3 world_pos;
    float3 meshlet_colour;
    float4 sv_position : SV_Position;
};

/*
[shader("vertex")]
VertexOutput vert_main(in float3 world) {
    VertexOutput output;

    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    float4 world_position = float4(world, 1.0);

    float4 t = mul(world_position, uniform_buffer.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    output.sv_position = t;
    output.world_pos = world_position.xyz;
    return output;
}
*/


struct DummyPayload {
    float3[3] verts;
    uint fixed_task_lod;
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

    int fixed_task_lod = 0;
    let fixed_header = tess_buffer.Load<Header>(4*4*fixed_task_lod);

    DummyPayload payload = DummyPayload(verts, fixed_task_lod);
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

    DispatchMesh(dispatch_mesh_size, 1, fixed_header.triangle_count, payload);
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
[numthreads(66, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[128], out indices uint3 triangles[128], uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, payload DummyPayload muh_payload) {
    // get buffers and preamble stuff (I <3 bindless)
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let tess_buffer = storage_buffers_readonly[3];
    float3[3] starting_verts = muh_payload.verts;   
    float3[3] verts = {};   
    
    // apply first order tesselation (emitted by task shader)
    let fixed_header = tess_buffer.Load<Header>(4*4*muh_payload.fixed_task_lod);
    uint3 fixed_tri = tess_buffer.Load<uint3>(fixed_header.triangle_array_byte_offset + id.z * 4 * 3);
    for (int i = 0; i < 3; i++) {
        float3 weights_fo = tess_buffer.Load<float3>(fixed_header.vertex_array_byte_offset + 4 * 3 * fixed_tri[i]);
        verts[i] = starting_verts[0] * weights_fo.x + starting_verts[1] * weights_fo.y + starting_verts[2] * weights_fo.z;
    }    
        
    // calculate LOD level based on distance to center of triangle
    float3 center = (verts[0] + verts[1] + verts[2]) / 3.0;
    float3 aabb_min = min3(verts[0], verts[1], verts[2]);
    float3 aabb_max = max3(verts[0], verts[1], verts[2]);

    // project aabb to clip space
    float4 clip_space_aabb_min = mul(float4(aabb_min, 1), uniforms.view_projection_matrix);
    float4 clip_space_aabb_max = mul(float4(aabb_max, 1), uniforms.view_projection_matrix);
    float4 size_clip_space = clip_space_aabb_max - clip_space_aabb_min;

    float screen_heuristic = length(size_clip_space.xy) * 25.0;
    float distance_heuristic = 1 - distance(center, uniforms.position.xyz) / 15.0;


    int lod = clamp(floor(screen_heuristic + distance_heuristic), 0, 3);
    //int lod = 3 - clamp(floor(distance(center, uniforms.position.xyz) / 10.0), 0, 3);
    
    // fetch header for LOD from CPU generated tesselation buffer
    let header = tess_buffer.Load<Header>(4*4*lod);
    
    // FIXME: this is a big waste of the 63 other threads in the case that we are just spawning in a non-tesselated triangle...
    // ideally this mesh shader would not even execute for non tesselated geometry
    if (gtid.x > header.vertex_count) {
        return;
    }
    
    // calculate world space and clip space position of tesselated vertex (per thread!)
    float3 weights = tess_buffer.Load<float3>(header.vertex_array_byte_offset + gtid.x * 4 * 3);
    float3 weighted_sum = verts[0] * weights.x + verts[1] * weights.y + verts[2] * weights.z;
    float4 world_position = float4(weighted_sum, 1);

    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    vertices[gtid.x].world_pos = world_position.xyz;
    vertices[gtid.x].sv_position = t;
    vertices[gtid.x].meshlet_colour = hash33(gtid);
    
    if (gtid.x < header.triangle_count) {
        uint3 tri = tess_buffer.Load<uint3>(header.triangle_array_byte_offset + gtid.x * 4 * 3);
        triangles[gtid.x] = tri;
    }
    
    SetMeshOutputCounts(header.vertex_count, header.triangle_count);
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