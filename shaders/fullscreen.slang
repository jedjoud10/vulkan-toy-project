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

bool dda2(
    float3 ray_dir,
    float3 ray_pos,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);

    bool hit = false;
    for (int i = 0; i < MAX_ITERS; i++) {
        if (d(floored_pos).x < 1) {
            return true;
            /*
            float3 point = floored_pos - ray_dir;
            capture_performance_query_trace_ray();
            for (; i < MAX_ITERS; i++) {
                capture_performance_sdf_call();

                let k = d(point).x;

                if (k < THRESHOLD_DISTANCE) {
                    return true;
                }

                point += ray_dir * k;
            }
            break;
            */
        }
    
        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    return hit;
}


[shader("compute")]
[numthreads(8, 8, 1)]
void prepass(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID) {
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
        capture_performance_sdf_call();

        let k = d2(point).x;

        if (k < max(THRESHOLD_DISTANCE * (dist+1) * 1.5, THRESHOLD_DISTANCE) || dist > MAX_DISTANCE) {
            break;
        }
        
        dist += k;
        point += ray_dir * k;
    }

    /*
    i = 0;
    float dist2 = 0;
    int i2 = 0;
    for (i2; i2 < MAX_ITERS; i2++) {
        capture_performance_sdf_call();

        let k = d(point).x;

        if (k < THRESHOLD_DISTANCE || dist > MAX_DISTANCE) {
            break;
        }
        
        dist2 += k;
        point += ray_dir * k;
    }
    */
    
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

    float3 col = 0;

    uint64_t clock = clockARB();
    float3 point = ray_pos + ray_dir * prepass.x;
    float dist = prepass.x;
    capture_performance_query_trace_ray();
    int i = 0;
    for (;i < MAX_ITERS; i++) {
        capture_performance_sdf_call();

        let k = d2(point);

        if (k.x < THRESHOLD_DISTANCE) {
            col = (float)i / MAX_ITERS;
            //col = k.yzw;
            //float3 normal = k.yzw;
            //float3 normal = 0.0;
            //col = clamp(dot(normal, uniforms.sun.xyz), 0, 1);
            
            break;
        }

        if (dist > MAX_DISTANCE || any(point < -16 | point > 16)) {
            break;
        }

        point += ray_dir * k.x;
        dist += k.x;
    }
    
    uint64_t clock2 = clockARB();
    //col = min((float64_t)(clock2-clock) / ((float64_t)100000.0), 1);
    //col = prepass.y;

    // col += sampled_images[SDF_TEXTURE4_SAMPLED_IMAGE_IDX].Sample(samplers[LINEAR_CLAMPED_SAMPLER_IDX], screen_uvs).z;
    // col = min(prepass.x, 100)/100;
    /*
    if (dda2(ray_dir, ray_pos)) {
        col = 1.0;
    }
    */

    /*
    if (uniforms.debug_type == 0) {
        if (is_bit_set(uniforms.toggles_bitmask, 0)) {
            col = trace_scene_multibounce(id.xy, ray_pos, ray_dir);
        } else {
            col = trace_scene_singlebounce(id.xy, ray_pos, ray_dir);
        }
    } else {
        uint64_t clock = clockARB();

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
            case 4: 
                uint64_t clock2 = clockARB();
                col = min((float64_t)(clock2-clock) / ((float64_t)100000.0), 1);
                break;
            case 5: 
                col = trace_vxgi(ray_pos, ray_dir, 0.0);
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

    float sdf = storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Load(uint3(id.xy, 2) / 10);
    //col = abs(fract(sdf));
    */

    //storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(lerp(last, col, lerp_factor), 1));
    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}