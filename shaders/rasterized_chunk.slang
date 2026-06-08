#language slang 2026
#include "utils/pbr.slang"
#include "utils/bindless.slang"
#include "utils/triplanar.slang"

struct VertexInput {
    float3 world;
    //float3 normal;
};

struct VertexOutput {
    float3 world_pos;
    //float3 normal;
    float3 colour;
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
    output.colour = (snoise(world_position.xyz * 0.1) * 0.5 + 0.5);
    //output.normal = input.normal;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    let uniforms = load_per_frame_uniforms();
    
    float3 world = in_vert.world_pos;


    float hwrt_shadow = 1.0;

    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));

    if (dot(normal, uniforms.sun.xyz) > 0) {
        hwrt_shadow = hw_trace_shadow(world);
    }

    //float3 normal = -normalize(in_vert.normal);
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);
    float3 diffuse = 1.0;
    float roughness = 0.8;
    float metallic = 0.0;
    float ao = hw_ao(world, normal);
    float shadow = (1.0-clouds(uniforms.sun.xyz, world).a) * hwrt_shadow; 



    float3 ray_pos = uniforms.position.xyz;

    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
    CameraData camera_data = CameraData(view);

    // approximate sun color based on height
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);


    /*
    desc.Origin = world;
    desc.Direction = reflect(-view, normal);
    desc.TMax = 100;
    desc.TMin = 0.001;

    sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);
    sq.Proceed();
    if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return float4(sq.CandidateRayBarycentrics(), 0, 1);
    }
    */

    if (uniforms.debug_type == 0) {
        return float4(lighting(pbr_surface_data, camera_data, sun_data), 1);
    } else if (uniforms.debug_type == 1) {
        return float4(normal, 1);
    } else if (uniforms.debug_type == 2) {
        return float4(view, 1);
    }

    return float4(1);
}