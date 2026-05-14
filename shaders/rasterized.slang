#language slang 2026
#include "pbr.slang"
#include "uniform_thing.slang"
#include "ray_stuff_other.slang"

[[vk::binding(0, 0)]]
ParameterBlock<UniformThing> uniform_buffer;

[[vk::binding(0, 1)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(1, 1)]]
StructuredBuffer<uint32_t> svo_indices_buffer;

[[vk::binding(2, 1)]]
StructuredBuffer<uint64_t> svo_aabbs_buffer;

[[vk::binding(3, 1)]]
SamplerCube<float4> skybox_sampler;

[[vk::binding(4, 1)]]
Sampler2D<float4> clouds_sampler;

[[vk::binding(5, 1)]]
Sampler3D<float4> svt_sampler;


struct VertexInput {
    float3 world;
};

struct VertexOutput {
    float3 world_pos;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;

    float4 world_position = float4(input.world, 1.0);

    float4 t = mul(mul(world_position, uniform_buffer.view_matrix), uniform_buffer.projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    output.sv_position = t;
    output.world_pos = world_position.xyz;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float3 normal = -normalize(cross(ddy(inVert.world_pos), ddx(inVert.world_pos)));
    float3 world = inVert.world_pos;
    float3 diffuse = 0.5;
    float roughness = 0.8;
    float metallic = 0.2;
    float ao = 1.0;
    float shadow = 1.0; 

    float3 ray_dir = normalize(uniform_buffer.position.xyz - world);
    float3 ray_pos = uniform_buffer.position.xyz;
    /*

    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
    CameraData camera_data = CameraData(ray_dir);

    // approximate sun color based on height
    float3 sun = uniform_buffer.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);
    SkyData sky_data = SkyData(skybox_sampler, clouds_sampler);

    float3 colour = lighting(pbr_surface_data, camera_data, sun_data, sky_data);

    return float4(colour, 1.0);
    */
    

    SparseVoxelOctree svo = SparseVoxelOctree(svo_indices_buffer, svo_bitmasks_buffer, svo_aabbs_buffer);
    Statistics stats = Statistics(0, 0);
    var ray = trace_shi<1024>(world + 0.5 + uniform_buffer.sun.xyz * 0.1, uniform_buffer.sun.xyz, svo, stats);

    return ray.hit ? 1.0 : 0.0;
}