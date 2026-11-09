#ifndef MS_COMMONS
#define MS_COMMONS

#include "utils/pbr.slang"
#include "utils/bindless.slang"

struct VertexOutput {
    float3 world_pos;
    float3 colour;
    float4 sv_position : SV_Position;
    float4 user;
};

VertexOutput transform_vertex(
    float3 vertex_ws,
    float3 colour
) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    float4 world_position = float4(vertex_ws, 1);

    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    VertexOutput vert;
    vert.world_pos = world_position.xyz;
    vert.sv_position = t;
    vert.colour = colour;
    vert.user = 0.0;
    return vert;
}

float4 shared_frag_shading(VertexOutput in_vert) {
    return 1.0;
    /*
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    
    float3 world = in_vert.world_pos;
    
    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));
    float3 view = normalize(in_vert.world_pos - uniforms.position.xyz);

    float3 diffuse = 0.5;
    float roughness = 0.8;
    float metallic = 0.2;
    float ao = 1.0;
    float shadow = 1.0; 

    float3 ray_pos = uniforms.position.xyz;

    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
    CameraData camera_data = CameraData(view);

    // approximate sun color based on height
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    float3 colour = lighting(pbr_surface_data, camera_data, sun_data);

    float3 reflection = sample_skybox_and_clouds(world, reflect(view, normal));
    float3 refraction = sample_skybox_and_clouds(world, refract(view, normal, 0.95));

    if (uniforms.debug_type == 0) {
        return float4(colour, 1);
    } else if (uniforms.debug_type == 1) {
        return float4(in_vert.colour, 1);
    } else if (uniforms.debug_type == 2) {
        return float4(reflection, 1);
    } else if (uniforms.debug_type == 3) {
        return float4(refraction, 1);
    } else {
        return float4(normal * 0.5 + 0.5, 1);
    }
    */
}

#endif