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

    [loop]
    for (; iter < MAX_DISTANCE; iter++) {
        float dist = d(ray_pos);

        if (dist < 0.001) {
            int16_t3 floored_pos = (int16_t3)floor(ray_pos * 10.0);
            float3 side_dist = ((dir_sign * (floored_pos - ray_pos * 10.0) + dir_sign * 0.5 + 0.5) * inv_dir);
            bool3 eqs = false;

            for (; iter < MAX_DISTANCE; iter++) {

                float distance_offset = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) / 10.0);
                float3 position = ray_pos + distance_offset * ray_dir;
                if (d(position) > 0.001) {
                    ray_pos = position + ray_dir * 0.001;
                    break;
                }

                /*
                uint8_t3 local = abs(floored_pos % 4);
                uint8_t bit_index = local.x | local.y << 2 | local.z << 4; 
                bool b = (raw >> bit_index) & 1 != 0;
                
                if (b) {
                    uint8_t face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
                    int8_t3 normal = get_face_normal(face, dir_sign);
                    return TestDdaTraceResults(iter, floored_pos, position, (float3)normal);
                }
                */

                if (d(round(position * 10) / 10.0) < 0) {
                    uint8_t face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
                    int8_t3 normal = get_face_normal(face, dir_sign);
                    return TestDdaTraceResults(iter, floored_pos, position, (float3)normal);
                }

                eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
                side_dist += select(eqs, inv_dir, 0.0);
                floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
            }
        }

        ray_pos += ray_dir * dist;






        
        
    }

    return none;

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

    Optional<TestDdaTraceResults> results = test_dda_trace<256>(ray_dir, ray_pos);
    if (results.hasValue) {
        col = results.value.iters / 64.0;
        // col = sin(results.value.block);
        // col = lighting2(PbrSurfaceData(results.value.position, 1.0, -results.value.normal, AoRoughnessMetallic(float3(1, 1, 0))), CameraData(-ray_dir), SunData(uniforms.sun.xyz, 1.0));
    } else {
        // col = sample_skybox_and_clouds(ray_pos, ray_dir);
    }

    
    storage_images[RENDERED_STORAGE_IMAGE_IDX].Store(id.xy, float4(col, 1));
}