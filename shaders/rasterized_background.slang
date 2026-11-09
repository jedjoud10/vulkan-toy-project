#language slang 2026
#include "utils/sky.slang"

#include "utils/bindless.slang"

static float2 positions[6] = float2[](
    float2(-1, -1), // top left
    float2(1, 1), // bottom right
    float2(-1, 1), // bottom left
    
    float2(-1, -1), // top left 
    float2(1, -1), // top right
    float2(1, 1) // bottom right
);

[shader("vertex")]
float4 vert_main(uint vertex_index: SV_VertexID) : SV_Position {
    float2 p = positions[vertex_index];
    float4 sv_position = float4(p, 1.0, 1.0);
    return sv_position;
}

[shader("fragment")]
float4 frag_main(float4 sv_position : SV_Position) : SV_Target {
    let uniform_data = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float2 screen_uvs = (sv_position.xy / uniform_data.screen_resolution);
    
    let ndc = float4(screen_uvs * 2 - 1, 1, 1);
    float3 ray_dir = mul(ndc, uniform_data.inv_projection_matrix).xyz;
    ray_dir = mul(float4(ray_dir.xy, 1, 0), uniform_data.inv_view_matrix).xyz;
    ray_dir = -ray_dir;

    float3 ray_pos = uniform_data.position.xyz;
    return float4(sample_skybox_and_clouds(ray_pos, ray_dir), 1.0);
    //return float4(get_environment_diffuse(ray_dir), 1.0);
}