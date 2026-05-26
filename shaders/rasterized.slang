#language slang 2026
#include <pbr.slang>
#include <uniform_thing.slang>
#include <ray_stuff_other.slang>
#include <bindless.slang>


struct VertexInput {
    float3 world;
};

struct VertexOutput {
    float3 world_pos;
    float3 meshlet_colour;
    float4 sv_position : SV_Position;
};

/*
[shader("vertex")]
VertexOutput vert_main(VertexInput input) {
    VertexOutput output;

    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    float4 world_position = float4(input.world, 1.0) + uint4(input.count % 100, 0, input.count / 100, 0);

    world_position.y += sin(input.count * 0.5 + uniform_buffer.time * 0) * 2;


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
void mesh_main(out vertices VertexOutput vertices[64], out indices uint3 triangles[98], uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    float2 meshlet_offset = (float2(id.x, id.y) * 7.0) * 10.0;
    float2 meshlet_offset_2 = (float2(id.x + 1, id.y + 1) * 7.0) * 10.0;
    float2 flat_center = (meshlet_offset+meshlet_offset_2) * 0.5;

    float3 meshlet_center = float3(flat_center.x, 0, flat_center.y); 

    let camera_frustum_planes = uniform_buffer.camera_frustum_planes;
    for (int f = 0; f < 6; f++) {
        float3 plane_normal = camera_frustum_planes[f].xyz;
        float plane_distance = camera_frustum_planes[f].w;

        if ((dot(plane_normal, meshlet_center) + plane_distance) < (-160.0 * 2.0)) {
            return;
        }
    }

    for (uint x = 0; x < 8; x++) {
        for (uint y = 0; y < 8; y++) {
            VertexOutput output;

            float2 p = float2(x, y) * 10 + meshlet_offset;
            float height = snoise(p * 0.008 + uniform_buffer.time * 0.4);
            
            float4 world_position = float4(p.x, height * 1, p.y, 1);

            float4 t = mul(world_position, uniform_buffer.view_projection_matrix);
            t.y = -t.y;
            t.x = -t.x;

            int index = y + x * 8;

            vertices[index].sv_position = t;
            vertices[index].world_pos = world_position.xyz;
            vertices[index].meshlet_colour = hash33(id);
        }
    }

    int quad_index = 0;
    for (uint x = 0; x < 7; x++) {
        for (uint y = 0; y < 7; y++) {
            uint4 corners = uint4(y + x * 8, y + x * 8 + 1, y + x * 8 + 8, y + x * 8 + 1 + 8);

            triangles[quad_index * 2] = uint3(corners.x, corners.y, corners.z);
            triangles[quad_index * 2 + 1] = uint3(corners.w, corners.z, corners.y);
            quad_index++;
        }
    } 

    SetMeshOutputCounts(64, quad_index*2);
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
    
    float3 reflection = sample_skybox_and_clouds(world, reflect(view, normal), clouds_sampler, skybox_sampler);

    return float4(reflection, 1.0);
}