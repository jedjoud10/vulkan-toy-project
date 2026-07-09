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
VertexOutput vert_main(VertexInput input) {
    VertexOutput output;

    let uniforms = load_per_frame_uniforms();

    float4 world_position = float4(input.world, 1.0);

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
float4 frag_main(VertexOutput in_vert, uniform uint material_base_texture_index) : SV_Target {
    let uniforms = load_per_frame_uniforms();

    if ((uniforms.toggles_bitmask & 8) == 8) {
        return 1.0;
    }

    float3 dp1 = ddx(in_vert.world_pos);
    float3 dp2 = ddy(in_vert.world_pos);
    float3 normal = normalize(cross(dp1, dp2));            
    float3 world = in_vert.world_pos;
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);  

    
    float3 diffuse = sample_triplanar(material_base_texture_index, world, normal);
    float3 arm = sample_triplanar(material_base_texture_index+1, world, normal);
    normal = normalize(sample_triplanar_normal(material_base_texture_index+2, world, normal));
    

    //float3 diffuse = 0.5 * in_vert.colour;
    //float3 arm = float3(1.0, 0.8, 0);
    
    

    
    // construct PBR data
    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, AoRoughnessMetallic(arm));
    CameraData camera_data = CameraData(view);
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    return float4(lighting3<1>(pbr_surface_data, camera_data, sun_data), 1);
}