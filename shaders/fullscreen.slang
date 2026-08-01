#language slang 2026
#include "utils/bindless.slang"
#include "utils/ray_stuff_other.slang"
#include "utils/sky.slang"
#include "utils/pbr.slang"

float d(float3 p) {
    return min(length(p - 10*round(p/10.0)) - 2, p.y);
}

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

    for (int i = 0; i < 256; i++) {
        let k = d(ray_pos);
        ray_pos += ray_dir * k;

        if (k < 0.01) {
            col = sin(ray_pos) * 0.5 + 0.5;
            break;
        }
    }

    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}