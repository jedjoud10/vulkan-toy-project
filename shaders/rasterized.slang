#language slang 2026
#include "pbr.slang"
#include "uniform_thing.slang"
#include "ray_stuff_other.slang"
#include "bindless.slang"


struct VertexInput {
    float3 world;
    uint count : SV_InstanceID;
};

struct VertexOutput {
    float3 world_pos;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
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

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    float3 view = normalize(inVert.world_pos - uniform_buffer.position.xyz);

    float3 normal = -normalize(cross(ddy(inVert.world_pos), ddx(inVert.world_pos)));
    float3 reflected = reflect(view, normal);
    return float4((sampled_images_cube[SKYBOX_SAMPLER_IMAGE_IDX]).Sample(reflected).xyz, 0.0);
}