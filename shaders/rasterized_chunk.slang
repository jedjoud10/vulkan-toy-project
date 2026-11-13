#language slang 2026
#include "utils/pbr.slang"
#include "utils/bindless.slang"
#include "utils/triplanar.slang"
#include "utils/materials.slang"

struct PushConstants {
    float3 *vertex_buffer_address;
    float3 *normal_buffer_address;
    uint *index_buffer_address;

    uint draw_index;
    uint triangle_count_mesh;
};

struct VertexOutput {
    float3 world_pos;
    float3 normal;
    float3 colour;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vert_main(uint index: SV_VertexID, uniform PushConstants pc) {
    VertexOutput output;

    float3 world = pc.vertex_buffer_address[index];
    float3 normal = pc.normal_buffer_address[index];

    let uniforms = load_per_frame_uniforms();

    float4 world_position = float4(world, 1.0);

    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    output.sv_position = t;
    output.world_pos = world_position.xyz;
    output.colour = (snoise(world_position.xyz * 0.1) * 0.5 + 0.5);
    output.normal = normal;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert, uniform PushConstants pc) : SV_Target {
    let uniforms = load_per_frame_uniforms();

    if ((uniforms.toggles_bitmask & 8) == 8) {
        return float4(hash31((float)pc.draw_index * 123.456), 1);
    }

    /*
    float3 dp1 = ddx(in_vert.world_pos);
    float3 dp2 = ddy(in_vert.world_pos);
    float3 normal = normalize(cross(dp1, dp2));
    */

    float3 normal = normalize(in_vert.normal);
    float3 world = in_vert.world_pos;
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);

    // bottom layer
    let base_layer = BaseEmptyLayer();
    let with_rock = add_layer(base_layer, Material.load(4), 1);

    // top layer
    let top_layer = BaseEmptyLayer();
    let with_sand = add_layer(top_layer, Material.load(6), 1.0);
    let with_puddles = add_layer(with_sand, Material.load(3), smoothstep(0.0, 0.3, snoise(world * 0.01)));

    // combine bottom and top with normal
    let combined = add_layer(with_rock, with_puddles, smoothstep(0.6, 0.96, normal.y));
    Layer res = smush(combined, world*3, normal);

    // construct PBR data
    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, res.diffuse, res.normal, AoRoughnessMetallic(res.arm));
    CameraData camera_data = CameraData(view);
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    // return float4(lighting3<1>(pbr_surface_data, camera_data, sun_data), 1);
    float3 rendered = lighting2(pbr_surface_data, camera_data, sun_data);
    // float3 rendered = hw_trace_shadow_light(world, sun);
    return float4(rendered, 1);
}