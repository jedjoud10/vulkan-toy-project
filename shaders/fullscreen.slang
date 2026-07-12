#language slang 2026
#include "utils/bindless.slang"
#include "utils/ray_stuff_other.slang"
#include "utils/sky.slang"
#include "utils/pbr.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) {


    let uniforms = load_per_frame_uniforms();
    float2 screen_uvs = (id.xy / uniforms.screen_resolution);
    
    let ndc = float4(screen_uvs * 2 - 1, 1, 1);
    float3 ray_dir = mul(ndc, uniforms.inv_projection_matrix).xyz;
    ray_dir = mul(float4(ray_dir.xy, 1, 0), uniforms.inv_view_matrix).xyz;
    ray_dir = -ray_dir;

    float3 ray_pos = uniforms.position.xyz;

    Statistics stats = Statistics();
    var ray = trace_shi<4096>(ray_pos, ray_dir, get_bindless_svo(), stats);


    float3 col = 0.0;
    if (ray.hit) {
        float3 diffuse = 1.0;
        //float3 diffuse = sampled_images[1].Sample(samplers[LINEAR_UNCAMPLED_SAMPLER_IDX], res.value.position.xz).xyz;
        float3 arm = float3(1.0, 0.5, 0.0);
        float3 normal = get_face_normal(ray.face, sign(ray_dir));
        float3 position = get_accurate_world_position_with_face2(ray.distance * ray_dir + ray_pos, ray.specific_block_axis_thing, ray.face, sign(ray_dir));
        
        // construct PBR data
        PbrSurfaceData pbr_surface_data = PbrSurfaceData(position, diffuse, normal, AoRoughnessMetallic(arm));
        CameraData camera_data = CameraData(-ray_dir);
        float3 sun = uniforms.sun.xyz;
        float3 sun_colour = sun_colour(sun.xyz);
        SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

        col = lighting3<1>(pbr_surface_data, camera_data, sun_data);
    } else {
        col = sample_skybox_and_clouds(ray_pos, ray_dir);
    }
    

    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}