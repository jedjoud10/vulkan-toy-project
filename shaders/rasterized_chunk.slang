#language slang 2026
#include "utils/pbr.slang"
#include "utils/bindless.slang"


struct VertexInput {
    float3 world;
    uint draw_chunk_index : SV_DrawIndex;
};

struct VertexOutput {
    float3 world_pos;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vert_main(VertexInput input, uniform int3 chunk_offset) {
    VertexOutput output;

    let uniforms = load_per_frame_uniforms();

    float4 world_position = float4(input.world + (float3)chunk_offset * 64, 1.0);

    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    output.sv_position = t;
    output.world_pos = world_position.xyz;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    let uniforms = load_per_frame_uniforms();
    
    float3 world = in_vert.world_pos;
    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);
    
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

    if (uniforms.debug_type == 0) {
        return float4(lighting(pbr_surface_data, camera_data, sun_data), 1);
    }

    return float4(1);
}