#language slang 2026
#include "utils/bindless.slang"
#include "utils/font.slang"

struct DebugTextLineHeader {
    uint32_t start_byte_offset;
    uint32_t char_count;
}
        
void render_debug_font_raw(uint2 id, inout float3 color, float3 font_colour) {
    // font scaling
    id /= 2;

    // the font we are using is 8x8
    uint2 pixel_offset_inside_char = id % 8;

    // do some flipping and convert to index in range [0..64)
    uint pixel_index_inside_char = pixel_offset_inside_char.x + pixel_offset_inside_char.y * 8; 
    
    uint char_index = id.x / 8;
    uint line_index = id.y / 8;

    
    let font_buffer = storage_buffers_readonly[STORAGE_BUFFER_DEBUG_TEXT_LINES];
    let total_num_lines = font_buffer.Load<uint32_t>(0);

    // only render the required number of lines
    if (line_index >= total_num_lines) {
        return;
    }

    // invert line order
    line_index = (total_num_lines-1) - line_index;

    // get header for this line
    // 4 is the byte size of uint32
    // 4 * 2 is the byte size of the header itself
    let header = font_buffer.Load<DebugTextLineHeader>(4 + line_index * 4 * 2); 

    // only render the required number of characters
    if (char_index >= header.char_count) {
        return;
    }

    // the CPU has already applied the byte offset to start_byte_offset
    uint64_t char = font8x8_basic[font_buffer.Load<uint8_t>(char_index + header.start_byte_offset)];

    if (((char >> pixel_index_inside_char) & 1) == 1) {
        color = font_colour;
    } else {
        color = 0.0;
    }
}

void render_debug_font(uint2 id, inout float3 color) {
    for (int x = -1; x < 2; x++) {
        for (int y = -1; y < 2; y++) {
            // massive performance regression on mesa radv compared to windows
            // render_debug_font_raw((int2)id + int2(x,y) * 2, color, all(int2(x, y) == 0) ? 1.0 : 0.0);
        }
    }


    

    render_debug_font_raw(id, color, 1.0);
}

[SpecializationConstant] const uint DOWNSCALE_FACTOR = 1;

// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
float3 aces(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}


//#define SHORT_CIRCUIT_NO_TONEMAPPING

[shader("compute")]
[numthreads(8, 8, 1)]
void write_swapchain_image(uint3 id: SV_DispatchThreadID) {
    let pre_compose = storage_images[RENDERED_STORAGE_IMAGE_IDX];
    let swapchain_image = storage_images[SWAPCHAIN_STORAGE_IMAGE_IDX];
    let entire_bloom_image = sampled_images[ENTIRE_BLOOM_SAMPLED_IMAGE_IDX];
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);


#ifdef SHORT_CIRCUIT_NO_TONEMAPPING
    swapchain_image[id.xy] = pre_compose[id.xy / DOWNSCALE_FACTOR];
    return;
#endif

    uint2 src_coords = id.xy / DOWNSCALE_FACTOR;
    uint2 dst_coords = id.xy;
    
    
    float2 screen_uvs_negative_one_to_one = ((float2)src_coords) / uniforms.screen_resolution;
    screen_uvs_negative_one_to_one *= 2.0;
    screen_uvs_negative_one_to_one -= 1.0;
    screen_uvs_negative_one_to_one.y = -screen_uvs_negative_one_to_one.y;
    screen_uvs_negative_one_to_one.x = -screen_uvs_negative_one_to_one.x;

    float2 screen_uvs = -screen_uvs_negative_one_to_one * 0.5 + 0.5;

    float3 final_colour = pre_compose[src_coords].xyz;
   
    SamplerState sampler = samplers[LINEAR_CLAMPED_SAMPLER_IDX];

    float bloom_strength = 0.05;
    float3 bloom = entire_bloom_image.SampleLevel(sampler, screen_uvs, 2.0).xyz * bloom_strength;

    float3 col = pow(aces(final_colour + bloom), 1/2.2);
    
    //float vignette = saturate(length(abs(screen_uvs_negative_one_to_one)) * 0.8 - 0.8);
    //col = lerp(col, 0.0, vignette);

    uint2 text_pixel_id = id.xy;
    // start at the bottom left instead...
    text_pixel_id.y = (uint)uniforms.screen_resolution.y * DOWNSCALE_FACTOR - text_pixel_id.y;
    render_debug_font(text_pixel_id, col);


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
    SamplerState sampler = samplers[LINEAR_CLAMPED_SAMPLER_IDX];
    

    float4 sum = 0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 offset = float2(x,y);
            float2 uvs = ((float2)(id.xy * 2) + offset * 2.0 + 1) / previous_bloom_mip_resolution;
            float4 loaded = previous_bloom_mip.Sample(sampler, uvs);
            
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
    SamplerState sampler = samplers[LINEAR_CLAMPED_SAMPLER_IDX];

    float4 sum = 0;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            float2 uvs = ((float2)(id.xy + int2(x,y))) * 0.5 / previous_bloom_mip_resolution;
            sum += previous_bloom_mip.Sample(sampler, uvs);
        }
    }
    
    if (!any(isnan(sum))) {
        next_bloom_mip[id.xy] = sum / 4.0;
    }
}
