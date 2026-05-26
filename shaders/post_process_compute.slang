#language slang 2026
#include <ray_stuff_other.slang>
#include <lighting.slang>
#include <pbr.slang>
#include <noises.slang>
#include <bindless.slang>
#include <font.slang>
#include <uniform_thing.slang>

[SpecializationConstant] const uint DOWNSCALE_FACTOR = 1;



//#define SHORT_CIRCUIT_NO_TONEMAPPING

[shader("compute")]
[numthreads(8, 8, 1)]
void write_swapchain_image(uint3 id: SV_DispatchThreadID) {
    let pre_compose = storage_images[RENDERED_STORAGE_IMAGE_IDX];
    let swapchain_image = storage_images[SWAPCHAIN_STORAGE_IMAGE_IDX];
    let entire_bloom_image = sampled_images[ENTIRE_BLOOM_SAMPLED_IMAGE_IDX];
    let uniform_thing = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

     

#ifdef SHORT_CIRCUIT_NO_TONEMAPPING
    swapchain_image[id.xy] = pre_compose[id.xy / DOWNSCALE_FACTOR];
    return;
#endif

    uint2 src_coords = id.xy / DOWNSCALE_FACTOR;
    uint2 dst_coords = id.xy;
    
    
    float2 screen_uvs_negative_one_to_one = ((float2)src_coords) / uniform_thing.screen_resolution;
    screen_uvs_negative_one_to_one *= 2.0;
    screen_uvs_negative_one_to_one -= 1.0;
    screen_uvs_negative_one_to_one.y = -screen_uvs_negative_one_to_one.y;
    screen_uvs_negative_one_to_one.x = -screen_uvs_negative_one_to_one.x;

    float2 screen_uvs = -screen_uvs_negative_one_to_one * 0.5 + 0.5;

    float3 final_colour = pre_compose[src_coords].xyz;

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
    
    float3 bloom = entire_bloom_image.SampleLevel(screen_uvs, 2.0).xyz;

    float3 col = pow(aces(final_colour + bloom * 0.1), 1/2.2);
    
    //float vignette = saturate(length(abs(screen_uvs_negative_one_to_one)) * 0.8 - 0.8);
    //col = lerp(col, 0.0, vignette);


    swapchain_image[dst_coords] = float4(col, 0);
}

// https://stackoverflow.com/questions/20746172/blur-an-image-using-3x3-gaussian-kernel
static const float GAUSSIAN_KERNEL_3X3[9] = {
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 8.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 16.0,
};

static const float MIN_BLOOM_THRESHOLD = 0.3;

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_downsample(uint3 id: SV_DispatchThreadID, uniform float2 previous_bloom_mip_resolution, uniform uint src_sampled_image_idx, uniform uint dst_storage_image_idx) {
    let previous_bloom_mip = sampled_images[src_sampled_image_idx];
    let next_bloom_mip = storage_images[dst_storage_image_idx];
    

    float4 sum = 0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 offset = float2(x,y);
            float2 uvs = ((float2)(id.xy * 2) + offset * 2.0 + 1) / previous_bloom_mip_resolution;
            float4 loaded = previous_bloom_mip.Sample(uvs);
            
            if (length(loaded) > MIN_BLOOM_THRESHOLD) {
                float kernel_strength = GAUSSIAN_KERNEL_3X3[(x+1) * 3 + (y+1)];
                sum += clamp(loaded * kernel_strength, 0, 1000);
            }
        }
    }

    next_bloom_mip[id.xy] = sum;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_upsample(uint3 id: SV_DispatchThreadID, uniform float2 previous_bloom_mip_resolution, uniform uint src_sampled_image_idx, uniform uint dst_storage_image_idx) {
    let previous_bloom_mip = sampled_images[src_sampled_image_idx];
    let next_bloom_mip = storage_images[dst_storage_image_idx];

    float4 sum = 0;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            float2 uvs = ((float2)(id.xy + int2(x,y))) * 0.5 / previous_bloom_mip_resolution;
            sum += previous_bloom_mip.Sample(uvs);
        }
    }
    
    if (!any(isnan(sum))) {
        next_bloom_mip[id.xy] = sum / 4.0;
    }
}