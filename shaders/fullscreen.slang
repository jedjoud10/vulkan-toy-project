#language slang 2026
#include "utils/bindless.slang"
#include "utils/sky.slang"
#include "utils/pbr.slang"

bool dda(
    float3 ray_dir,
    float3 ray_pos,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);

    bool hit = false;
    for (int i = 0; i < 32; i++) {
        if (d(floored_pos).x < 1) {
            hit = true;
            break;
        }
    
        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    return hit;
}

// TODO: figure out how we can optimize prepass with chunks that have been unloaded
// currently, we use the sample the global, uncached SDF, but that is super slow
// the mspf gains that we save in the main pass are lost due to the slowness of sampling the global sdf in the prepass. lmfao
// #define USE_PREPASS

[shader("compute")]
[numthreads(8, 8, 1)]
void prepass(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID) {
    #ifndef USE_PREPASS
    return;
    #endif

    uniforms = load_per_frame_uniforms();
    float2 screen_uvs = ((id.xy+0.5) / (uniforms.screen_resolution/4));
    
    let ndc = float4(screen_uvs * 2 - 1, 1, 1);
    float3 ray_dir = mul(ndc, uniforms.inv_projection_matrix).xyz;
    ray_dir = mul(float4(ray_dir.xy, 1, 0), uniforms.inv_view_matrix).xyz;
    ray_dir = normalize(-ray_dir);
    
    float3 ray_pos = uniforms.position.xyz;
    float3 point = ray_pos;
    float dist = 0;
    capture_performance_query_trace_ray();
    int i = 0;
    for (i; i < MAX_ITERS; i++) {
        let k = d(point).x;

        if (k < max(THRESHOLD_DISTANCE_PREPASS * (dist+1) * 1.5, THRESHOLD_DISTANCE_PREPASS) || dist > MAX_DISTANCE) {
            break;
        }
        
        dist += k;
        point += ray_dir * k;
    }
    
    storage_images[PRE_PASS_TEXTURE].Store(id.xy, float4(dist, (float)i / MAX_ITERS, 1, 1));
}

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
    float4 prepass = storage_images[PRE_PASS_TEXTURE].Load(id.xy / 4);

    /*

    bool hit = false;
    float3 hit_position = 0.0;
    float3 hit_normal = 0.0;
    float3 col = 0;

    uint64_t clock = clockARB();
#ifdef USE_PREPASS
    float starting_distance = prepass.x;
#else    
    float starting_distance = 0;
#endif
    
    // float starting_distance = max(prepass.x, trace_get_closest_to_surface(ray_pos, ray_dir)-0.1);
    float3 point = ray_pos + ray_dir * starting_distance;
    float dist = starting_distance;
    capture_performance_query_trace_ray();
    int total_ray_march_iters = 0;
    float closest_distance_to_surface = 1;
    
#ifdef OUTER_DDA_TRAVERSAL
    float3 modified_ray_pos = point / CHUNK_PHYSICAL_SIZE;
    int3 floored_pos = floor(modified_ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    int3 eqs;
    float3 side_dist = (floored_pos - modified_ray_pos + 0.5 + 0.5 * dir_sign);

    outer: for (int i = 0; i < DDA_TRAVERSAL_CHUNKS_ITER; i++) {

        uint chunk_start_index = storage_images_3d_uint[CHUNK_LOOKUP_TEXTURE].Load(floored_pos + (int)CHUNK_LOOKUP_MEGA_TEXTURE_HALF_SIZE);
        
        // make sure texel points to an actual chunk that has been loaded/generated
        if (chunk_start_index != uint.maxValue) {
            float3 test = (floored_pos - modified_ray_pos + 0.5 - 0.5 * dir_sign) * inv_dir;
            float d = max3(test.x, test.y, test.z);
            float3 point = (modified_ray_pos + ray_dir * max(d, 0)) * CHUNK_PHYSICAL_SIZE;
            point = clamp(point, floored_pos * CHUNK_PHYSICAL_SIZE + CHUNK_BOUNDARY_OFFSET_EPS, (floored_pos+1) * CHUNK_PHYSICAL_SIZE - CHUNK_BOUNDARY_OFFSET_EPS);
            
            inner: for (int o = 0;o < DDA_TRAVERSAL_RAY_MARCH_ITER; o++) {
                // check if we moved from one chunk to another chunk
                // if we did, then we must stop raymarching the current chunk
                if (any((floor(point / CHUNK_PHYSICAL_SIZE) != (float3)floored_pos))) {
                    break inner;
                }

                let sampled = sample_chunk(point, chunk_start_index);
                closest_distance_to_surface = min(closest_distance_to_surface, sampled.x);

                // check if we got close enough to the surface of the SDF
                if (sampled.x < THRESHOLD_DISTANCE) {
                    hit = true;
                    hit_position = point;  
                    hit_normal = sampled.yzw;          
                    break outer;
                }
            
                point += ray_dir * sampled.x;
                total_ray_march_iters++;

            }
        } 
    
        float3 reconst = side_dist * inv_dir;
        eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }
#else
    for (int i = 0; i < MAX_ITERS; i++) {
        let k = d2(point);

        if (k.x < THRESHOLD_DISTANCE) {
            hit = true;
            hit_position = point;  
            hit_normal = get_normal_test(point);          
            break;
        }

        if (dist > MAX_DISTANCE) {
            break;
        }

        point += ray_dir * k.x;
        dist += k.x;
        total_ray_march_iters++;
    }
#endif
    */

    /*
    if (hit) {
        col = dot(hit_normal, uniforms.sun.xyz) * floor_material(hit_position, hit_normal).diffuse;
    } else {
        // glow effect!
        col += saturate(0.2-closest_distance_to_surface)*0.6;
    }

    uint64_t clock2 = clockARB();
    */
    // col = min((float64_t)(clock2-clock) / ((float64_t)800000.0), 1);
    //col = prepass.y;

    // col += sampled_images[SDF_TEXTURE4_SAMPLED_IMAGE_IDX].Sample(samplers[LINEAR_CLAMPED_SAMPLER_IDX], screen_uvs).z;
    // col = min(prepass.x, 100)/100;
    /*
    if (dda2(ray_dir, ray_pos)) {
        col = 1.0;
    }
    */
    /*
    switch (uniforms.debug_type) {
        case 1: 
            col = (float)total_ray_march_iters / MAX_ITERS;
            break;
        case 2: 
            col = prepass.x;
            break;
        case 3: 
            col = prepass.y;
            break;
        case 4: 
            col = min((float64_t)(clock2-clock) / ((float64_t)800000.0), 1);
            break;
        case 5: 
            col = hit_normal;    
            // col = (float)sampled_from_cached_sdf / (sampled_from_cached_sdf + sampled_from_raw_sdf);
            break;
        default:
            break; 
    }
    */

    /*

    float lerp_factor = 1.0;
    if ((uniforms.toggles_bitmask & 2) != 2) {
        lerp_factor = 0.03;
    }

    if (any(isnan(last))) {
        last = 0;
    }

    float sdf = storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Load(uint3(id.xy, 2) / 10);
    //col = abs(fract(sdf));
    */

    //storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(lerp(last, col, lerp_factor), 1));
    // float3 col = trace_scene_multibounce(id.xy, ray_pos, ray_dir);
    float3 col = trace_scene_singlebounce(id.xy, ray_pos, ray_dir);
    
    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}