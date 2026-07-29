#language slang 2026
#include "utils/bindless.slang"
#include "utils/ray_stuff_other.slang"
#include "utils/sky.slang"
#include "utils/pbr.slang"


struct TestDdaTraceResults {
    uint iters;
    int3 block;
    float3 position;
    float3 normal;
}


float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float d(float3 p) {
    // return min(sdBox(p - float3(0, 3 * 20.0, 0), 5.0), p.y);
    return min(length(p - float3(0, 3 * 20.0, 0)) - 5.0, p.y) + sin(p.x * 0.05) * 5;
}


Optional<TestDdaTraceResults> test_dda_trace<let MAX_DISTANCE : int>(
    float3 ray_dir,
    float3 ray_pos,
) {
    half3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);  
    
    
    uint64_t raw = storage_buffers_readonly[STORAGE_BUFFER_TESTING_BUFFER].LoadAligned<uint64_t>(0);
    uint iter = 0;
    int wowza = 0;

    float3 position = ray_pos;
    float tmax = 0;

    /*
    for (; iter < MAX_DISTANCE; iter++) {

        int3 voxel_position = floor(position);
        Data k = start_from_root_go_down(voxel_position, wowza);

        if (k.leaf) {
            return TestDdaTraceResults(wowza + iter, voxel_position, 0, 0);
        }

        // int3 bounds_min = voxel_position;
        // int3 bounds_max = voxel_position+1;       

        // int3 bounds_min = k.min_bounds;
        // int3 bounds_max = k.max_bounds+1;    

        ray_box_intersection(inv_dir, dir_sign, ray_pos, bounds_min, bounds_max, tmax);
        

        

        
        position = ray_pos + ray_dir * (tmax + 0.01);
    }

    return none;
    */
    /*
    [loop]
    for (uint i = 0; i < MAX_DISTANCE; i++) {
        float dist = min(distance(ray_pos, 0.0) - 15  + sin(ray_pos.y), ray_pos.y);
        ray_pos += ray_dir * dist;

        if (dist < 0.01) {
            return TestDdaTraceResults(ray_pos, ray_pos, 1.0);
        }

    }

    return none;
    */
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

    /*
    float tmin = 0;
    bool hit = ray_box_intersection(1.0 / abs(ray_dir), sign(ray_dir), ray_pos, 0, 100, tmin);

    if (hit) {
        col = tmin;
    }
    */
    
    Statistics stats = Statistics();
    DdaTraversalOutput output = trace_shi(ray_pos, ray_dir, get_bindless_svo(), stats);

    if (output.hit) {
        if ((uniforms.toggles_bitmask & 1) == 1) {
            col = stats.iterations / 256.0;
        } else {
            col = 1.0;
        }
    }

    /*
    Optional<TestDdaTraceResults> results = test_dda_trace<256>(ray_dir, ray_pos);
    if (results.hasValue) {
        if ((uniforms.toggles_bitmask & 1) == 1) {
            col = results.value.iters / 64.0;
        } else {
            col = sin(results.value.block);
        }
        
        // col = lighting2(PbrSurfaceData(results.value.position, 1.0, -results.value.normal, AoRoughnessMetallic(float3(1, 1, 0))), CameraData(-ray_dir), SunData(uniforms.sun.xyz, 1.0));
    } else {
        // col = sample_skybox_and_clouds(ray_pos, ray_dir);
    }
    */

    
    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}