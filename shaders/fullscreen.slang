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
    ray_dir = normalize(-ray_dir);
    float3 ray_pos = uniforms.position.xyz;
    float3 col = 0.0;

    Configuration config = Configuration();
    Statistics stats = Statistics();
    DdaTraversalOutput output = trace_shi(ray_pos, ray_dir, config, get_bindless_svo(), stats);

    if (output.hit) {
        float3 world = ray_pos + ray_dir * output.distance;
        //world = get_accurate_world_position_with_face2(world, output.specific_block_axis_thing, output.face, sign(ray_dir));
        float3 normal = get_face_normal(output.face, sign(ray_dir));

        float3 albedo = snoise(world * 1.001) * 0.3 + 0.7;

        PbrSurfaceData surface = PbrSurfaceData(world, albedo, normal, AoRoughnessMetallic(float3(1, 0.9, 0)));
        SunData sun = SunData(uniforms.sun.xyz, 1.0);
        CameraData cam = CameraData(-ray_dir);

        col = lighting2(surface, cam, sun);
    } else {
        col = sample_skybox_and_clouds(ray_pos, ray_dir);
    }

    if ((uniforms.toggles_bitmask & 1) == 1) {
        col = stats.iterations / 256.0;
    }
    
    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}