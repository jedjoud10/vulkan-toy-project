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

    
    uint64_t clock = clockARB();
    TraceRes res = trace(ray_pos, ray_dir);
    uint64_t clock2 = clockARB();
    float3 col = 0;
    
    
    switch (uniforms.debug_type) {
        case 0: 
            col = (float)res.iters / MAX_ITERS;
            break;
        case 1: 
            col = min((float64_t)(clock2-clock) / ((float64_t)800000.0), 1);
            break;
        case 2: 
            col = res.normal;
            break;
        case 3: 
            col = res.position - floor(res.position);    
            break;
        case 4: 
            col = (1-res.closest_distance);    
            break;
        default:
            break; 
    }

    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}