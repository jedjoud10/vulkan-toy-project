#language slang 2026
#include <ray_stuff_other.slang>
#include <lighting.slang>
#include <pbr.slang>
#include <noises.slang>

[[vk::binding(0, 0)]]
readonly RWTexture2D<float4> pre_compose;

[[vk::binding(1, 0)]]
RWTexture2D<float4> render_target;

[shader("compute")]
[numthreads(8, 8, 1)]
void write_render_texture(uint3 id: SV_DispatchThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform uint debug_type, uniform float time) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    float4 pre_post_process = pre_compose[id.xy];
    
    float3 final_colour = pre_post_process.xyz;

    /*
    float total = 1;
    if (pre_post_process.w > 0.3) {
        float blur_size = pre_post_process.w;

        for (int x = -4; x <= 4; x++) {
            for (int y = -4; y <= 4; y++) {
                if (x != 0 && y != 0) {
                    float strength = 1/length(int2(x,y));
                    total += strength;

                    float4 sample = pre_compose[id.xy + int2(x,y) * blur_size];

                    if (sample.w > 0.3) {
                        final_colour += sample.xyz * strength;
                    }
                }
            }    
        }
        
        final_colour /= total;
    }
    */
    

    float3 col = pow(aces(final_colour), 1/2.2);
    float vignette = saturate(length(abs(screen_uvs)) * 0.8 - 0.8);
    col = lerp(col, 0.0, vignette);


    render_target[id.xy] = float4(col, 0);
}