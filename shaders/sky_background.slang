#language slang 2026
#include "sky.slang"
#include "uniform_thing.slang"
#include "bindless.slang"

struct VertexOutput {
    float2 uvs;
    float4 sv_position : SV_Position;
};

static float2 positions[6] = float2[](
    float2(-1, -1), // top left
    float2(1, 1), // bottom right
    float2(-1, 1), // bottom left
    
    float2(-1, -1), // top left 
    float2(1, -1), // top right
    float2(1, 1) // bottom right
);



[shader("vertex")]
VertexOutput vertMain(uint vertex_index: SV_VertexID) {
    VertexOutput output;
    float2 p = positions[vertex_index];
    output.sv_position = float4(p, 1.0, 1.0);
    output.uvs = p;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float2 screen_uvs = inVert.uvs;
    

    let ndc = float4(screen_uvs, 1, 1);
    let uniform_data = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float3 ray_dir = mul(ndc, uniform_data.inv_projection_matrix).xyz;
    ray_dir = mul(float4(ray_dir.xy, 1, 0), uniform_data.inv_view_matrix).xyz;
    ray_dir = -ray_dir;

    float3 ray_pos = uniform_data.position.xyz;
    return float4(sample_skybox_and_clouds(ray_pos, ray_dir, sampled_images[CLOUDS_SAMPLER_IMAGE_IDX], sampled_images_cube[SKYBOX_SAMPLER_IMAGE_IDX]), 1.0);
}