#language slang 2026
#include "utils/bindless.slang"
#include "utils/ray_stuff_other.slang"
#include "utils/sky.slang"
#include "utils/pbr.slang"


[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID) {
    uniforms = load_per_frame_uniforms();
    float2 screen_uvs = (id.xy / uniforms.screen_resolution);
    
    let ndc = float4(screen_uvs * 2 - 1, 1, 1);
    float3 ray_dir = mul(ndc, uniforms.inv_projection_matrix).xyz;
    ray_dir = mul(float4(ray_dir.xy, 1, 0), uniforms.inv_view_matrix).xyz;
    ray_dir = normalize(-ray_dir);
    
    float3 ray_pos = uniforms.position.xyz;
    float3 last = storage_images[RENDERED_STORAGE_IMAGE_IDX].Load(id.xy).xyz;

    float3 col = 0;

    if (uniforms.debug_type == 0) {
        col = trace_scene_multibounce(id.xy, ray_pos, ray_dir);
    } else {
        TraceRes res = trace(ray_pos, ray_dir);
    
        switch (uniforms.debug_type) {
            case 1: 
                col = res.iters / 256.0;
                break;
            case 2: 
                col = max(res.normal, 0);
                break;
            case 3: 
                col = res.sdf_calls / 256.0;
                break;
            default:
                col = 0.0; 
        }
    }


    float lerp_factor = 1.0;
    if ((uniforms.toggles_bitmask & 2) != 2) {
        lerp_factor = 0.03;
    }

    if (any(isnan(last))) {
        last = 0;
    }

    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(lerp(last, col, lerp_factor), 1));
}