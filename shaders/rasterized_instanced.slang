#language slang 2026
#include "utils/pbr.slang"
#include "utils/triplanar.slang"
#include "utils/bindless.slang"

struct PushConstants {
    float4x4 object_to_world;
    uint material_base_sampled_image_index;
    uint _padding;

    float3* vertex_buffer_address;
    float3* normal_buffer_address;
    float2 *uv_buffer_address;
    float3 *instance_buffer_address;
};

struct VertexOutput {
    float3 world_pos;
    float3 world_normal;
    float2 uv;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vert_main(uint index: SV_VertexID, uint instance: SV_InstanceID, uniform PushConstants pc) {
    VertexOutput output;

    let uniforms = load_per_frame_uniforms();

    float3 local_position = pc.vertex_buffer_address[index];
    float3 local_normal = pc.normal_buffer_address[index];
    float2 uv = pc.uv_buffer_address[index];
    float3 instance_offset = pc.instance_buffer_address[instance];

    float4 world_position = mul(float4(local_position + instance_offset, 1), pc.object_to_world);
    float4 world_normal = mul(float4(local_normal, 0), pc.object_to_world);
    


    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    output.sv_position = t;
    output.world_pos = world_position.xyz;
    output.world_normal = world_normal.xyz;
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert, uniform PushConstants pc) : SV_Target {
    let uniforms = load_per_frame_uniforms();

    // I eventually need to understand why this works but I don't...
    // compute tangent and bitangent from derivatives of position and UV
    float3 dp1 = ddx(in_vert.world_pos);
    float3 dp2 = ddy(in_vert.world_pos);
    float2 duv1 = ddx(in_vert.uv);
    float2 duv2 = ddy(in_vert.uv);

    // solve for tangent and bitangent
    float3 normal = normalize(cross(dp1, dp2));
    float3 tangent = normalize(duv2.y * dp1 - duv1.y * dp2);
    float3 bitangent = normalize(-duv2.x * dp1 + duv1.x * dp2);
    float3x3 tbn = float3x3(tangent, bitangent, normal);
            
    float3 world = in_vert.world_pos;
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);  

    if ((uniforms.toggles_bitmask & 1) == 1) {
        normal = normalize(in_vert.world_normal);
    }

    // sample textures
    let sampler = samplers[NEAREST_SAMPLER_IDX];
    float3 diffuse = sampled_images[pc.material_base_sampled_image_index].Sample(sampler, in_vert.uv).xyz;
    float3 arm = sampled_images[pc.material_base_sampled_image_index+1].Sample(sampler, in_vert.uv).xyz;
    
    // calculate world space normal perturbed by sampled tangent normal map
    if ((uniforms.toggles_bitmask & 2) == 2) {
        // texture encoded using bc5, only have the RG channels, reconstruct the B channel
        float2 tangent_normal = sampled_images[pc.material_base_sampled_image_index+2].Sample(sampler, in_vert.uv).xy * 2.0 - 1.0;
        
        // 1 - x^2 + y^2 = z^2 
        float reconstructed_z = sqrt(1 - (pow(tangent_normal.x, 2) + pow(tangent_normal.y, 2)));
        normal = normalize(mul(float3(tangent_normal, reconstructed_z), tbn)); 
    }

    // construct PBR data
    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, AoRoughnessMetallic(arm));
    CameraData camera_data = CameraData(view);
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    // return float4(lighting3<1>(pbr_surface_data, camera_data, sun_data), 1);
    float3 rendered = lighting2(pbr_surface_data, camera_data, sun_data);
    return float4(rendered, 1);
}