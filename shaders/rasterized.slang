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


/*
struct DummyPayload {
    uint2 offset;
}
// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("task")]
[numthreads(1, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    DummyPayload localPayload;
    localPayload.offset = id.xy * 7;
    if (all(gtid == 0)) {
    }
    DispatchMesh(1, 1, 1, localPayload);
}
*/

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[64], out indices uint3 triangles[128], uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    let vertex_buffer = storage_buffers_readonly[1];
    let index_buffer = storage_buffers_readonly[2];

    uint3 loaded_triangle = index_buffer.Load3(id.x * 4 * 3);
    float3[6] vs = {};
    //printf("%d %d %d", loaded_triangle.x, loaded_triangle.y, loaded_triangle.z);
        
    for (int i = 0; i < 3; i++) {
        float3 loaded_vertex = vertex_buffer.Load<float3>(loaded_triangle[i] * 4 * 3);
        vs[i] = loaded_vertex;
    }

    float3 t1 = (vs[0]+vs[1])*0.5;
    float3 t2 = (vs[0]+vs[2])*0.5;
    float3 t3 = (vs[1]+vs[2])*0.5;

    vs[3] = t1;
    vs[4] = t2;
    vs[5] = t3;

    triangles[0] = uint3(0, 3, 4);
    triangles[1] = uint3(4, 3, 5);
    triangles[2] = uint3(5, 3, 1);
    triangles[3] = uint3(2, 4, 5);

    for (int i = 0; i < 6; i++) {
        //printf("%f %f %f", vertex.x, vertex.y, vertex.z);
        float4 world_position = float4(vs[i]+hash33(vs[i]) * 0.1, 1);

        float4 t = mul(world_position, uniforms.view_projection_matrix);
        t.y = -t.y;
        t.x = -t.x;

        vertices[i].world_pos = world_position.xyz;
        vertices[i].sv_position = t;
        vertices[i].meshlet_colour = hash33(vs[i]);
    }
    
    
    SetMeshOutputCounts(6, 4);
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
    
    //float3 reflection = sample_skybox_and_clouds(world, reflect(view, normal), clouds_sampler, skybox_sampler);

    return float4(in_vert.meshlet_colour, 1);
}