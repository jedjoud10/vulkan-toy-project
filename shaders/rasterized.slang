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

const static uint[4] TESS_VERT_COUNT = {3,6,18,66, };
const static uint[4] TESS_TRI_COUNT = {1,4,16,64, };
const static float3[66][4] TESS_VERTICES_WEIGHTS = {{ float3(1, 0, 0),  float3(0, 1, 0),  float3(0, 0, 1), },{ float3(1, 0, 0),  float3(0, 1, 0),  float3(0, 0, 1),  float3(0.5, 0.5, 0),  float3(0.5, 0, 0.5),  float3(0, 0.5, 0.5), },{ float3(1, 0, 0),  float3(0, 1, 0),  float3(0, 0, 1),  float3(0.5, 0.5, 0),  float3(0.5, 0, 0.5),  float3(0, 0.5, 0.5),  float3(0.75, 0.25, 0),  float3(0.75, 0, 0.25),  float3(0.5, 0.25, 0.25),  float3(0.5, 0.25, 0.25),  float3(0.25, 0.25, 0.5),  float3(0.25, 0.5, 0.25),  float3(0.25, 0.5, 0.25),  float3(0, 0.75, 0.25),  float3(0.25, 0.75, 0),  float3(0.25, 0, 0.75),  float3(0, 0.25, 0.75),  float3(0.25, 0.25, 0.5), },{ float3(1, 0, 0),  float3(0, 1, 0),  float3(0, 0, 1),  float3(0.5, 0.5, 0),  float3(0.5, 0, 0.5),  float3(0, 0.5, 0.5),  float3(0.75, 0.25, 0),  float3(0.75, 0, 0.25),  float3(0.5, 0.25, 0.25),  float3(0.5, 0.25, 0.25),  float3(0.25, 0.25, 0.5),  float3(0.25, 0.5, 0.25),  float3(0.25, 0.5, 0.25),  float3(0, 0.75, 0.25),  float3(0.25, 0.75, 0),  float3(0.25, 0, 0.75),  float3(0, 0.25, 0.75),  float3(0.25, 0.25, 0.5),  float3(0.875, 0.125, 0),  float3(0.875, 0, 0.125),  float3(0.75, 0.125, 0.125),  float3(0.75, 0.125, 0.125),  float3(0.625, 0.125, 0.25),  float3(0.625, 0.25, 0.125),  float3(0.625, 0.25, 0.125),  float3(0.5, 0.375, 0.125),  float3(0.625, 0.375, 0),  float3(0.625, 0, 0.375),  float3(0.5, 0.125, 0.375),  float3(0.625, 0.125, 0.25),  float3(0.5, 0.125, 0.375),  float3(0.375, 0.125, 0.5),  float3(0.375, 0.25, 0.375),  float3(0.375, 0.25, 0.375),  float3(0.25, 0.375, 0.375),  float3(0.375, 0.375, 0.25),  float3(0.375, 0.375, 0.25),  float3(0.375, 0.5, 0.125),  float3(0.5, 0.375, 0.125),  float3(0.125, 0.375, 0.5),  float3(0.125, 0.5, 0.375),  float3(0.25, 0.375, 0.375),  float3(0.125, 0.5, 0.375),  float3(0, 0.625, 0.375),  float3(0.125, 0.625, 0.25),  float3(0.125, 0.625, 0.25),  float3(0.125, 0.75, 0.125),  float3(0.25, 0.625, 0.125),  float3(0.25, 0.625, 0.125),  float3(0.375, 0.625, 0),  float3(0.375, 0.5, 0.125),  float3(0, 0.875, 0.125),  float3(0.125, 0.875, 0),  float3(0.125, 0.75, 0.125),  float3(0.125, 0, 0.875),  float3(0, 0.125, 0.875),  float3(0.125, 0.125, 0.75),  float3(0.125, 0.125, 0.75),  float3(0.125, 0.25, 0.625),  float3(0.25, 0.125, 0.625),  float3(0.25, 0.125, 0.625),  float3(0.375, 0.125, 0.5),  float3(0.375, 0, 0.625),  float3(0, 0.375, 0.625),  float3(0.125, 0.375, 0.5),  float3(0.125, 0.25, 0.625), }, };
const static uint3[64][4] TESS_TRIANGLES = {{ uint3(0, 1, 2), },{ uint3(0, 3, 4),  uint3(4, 3, 5),  uint3(5, 3, 1),  uint3(2, 4, 5), },{ uint3(0, 6, 7),  uint3(7, 6, 8),  uint3(8, 6, 3),  uint3(4, 7, 8),  uint3(4, 9, 10),  uint3(10, 9, 11),  uint3(11, 9, 3),  uint3(5, 10, 11),  uint3(5, 12, 13),  uint3(13, 12, 14),  uint3(14, 12, 3),  uint3(1, 13, 14),  uint3(2, 15, 16),  uint3(16, 15, 17),  uint3(17, 15, 4),  uint3(5, 16, 17), },{ uint3(0, 18, 19),  uint3(19, 18, 20),  uint3(20, 18, 6),  uint3(7, 19, 20),  uint3(7, 21, 22),  uint3(22, 21, 23),  uint3(23, 21, 6),  uint3(8, 22, 23),  uint3(8, 24, 25),  uint3(25, 24, 26),  uint3(26, 24, 6),  uint3(3, 25, 26),  uint3(4, 27, 28),  uint3(28, 27, 29),  uint3(29, 27, 7),  uint3(8, 28, 29),  uint3(4, 30, 31),  uint3(31, 30, 32),  uint3(32, 30, 9),  uint3(10, 31, 32),  uint3(10, 33, 34),  uint3(34, 33, 35),  uint3(35, 33, 9),  uint3(11, 34, 35),  uint3(11, 36, 37),  uint3(37, 36, 38),  uint3(38, 36, 9),  uint3(3, 37, 38),  uint3(5, 39, 40),  uint3(40, 39, 41),  uint3(41, 39, 10),  uint3(11, 40, 41),  uint3(5, 42, 43),  uint3(43, 42, 44),  uint3(44, 42, 12),  uint3(13, 43, 44),  uint3(13, 45, 46),  uint3(46, 45, 47),  uint3(47, 45, 12),  uint3(14, 46, 47),  uint3(14, 48, 49),  uint3(49, 48, 50),  uint3(50, 48, 12),  uint3(3, 49, 50),  uint3(1, 51, 52),  uint3(52, 51, 53),  uint3(53, 51, 13),  uint3(14, 52, 53),  uint3(2, 54, 55),  uint3(55, 54, 56),  uint3(56, 54, 15),  uint3(16, 55, 56),  uint3(16, 57, 58),  uint3(58, 57, 59),  uint3(59, 57, 15),  uint3(17, 58, 59),  uint3(17, 60, 61),  uint3(61, 60, 62),  uint3(62, 60, 15),  uint3(4, 61, 62),  uint3(5, 63, 64),  uint3(64, 63, 65),  uint3(65, 63, 16),  uint3(17, 64, 65), }, };
// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(66, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[128], out indices uint3 triangles[128], uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    let vertex_buffer = storage_buffers_readonly[1];
    let index_buffer = storage_buffers_readonly[2];
    
    uint3 loaded_triangle = index_buffer.Load3(gid.x * 4 * 3);
    float3[3] verts = {};
    
    for (int i = 0; i < 3; i++) {
        float3 loaded_vertex = vertex_buffer.Load<float3>(loaded_triangle[i] * 4 * 3);
        verts[i] = loaded_vertex;
    }    

    
    float3 center = (verts[0] + verts[1] + verts[2]) / 3.0;
    //int lod = 3 - clamp(floor(distance(center, uniforms.position.xyz) / 100.0), 0, 3);
    int lod = 3;

    if (gtid.x > TESS_VERT_COUNT[lod]) {
        return;
    }
    
    //printf("%f %f %f", vertex.x, vertex.y, vertex.z);
    float3 weights = TESS_VERTICES_WEIGHTS[lod][gtid.x];
    float3 weighted_sum = verts[0] * weights.x + verts[1] * weights.y + verts[2] * weights.z;
    float4 world_position = float4(weighted_sum, 1);

    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    vertices[gtid.x].world_pos = world_position.xyz;
    vertices[gtid.x].sv_position = t;
    vertices[gtid.x].meshlet_colour = hash33(gid);

    if (gtid.x < TESS_TRI_COUNT[lod]) {
        triangles[gtid.x] = TESS_TRIANGLES[lod][gtid.x];
    }
    
    SetMeshOutputCounts(TESS_VERT_COUNT[lod], TESS_TRI_COUNT[lod]);
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