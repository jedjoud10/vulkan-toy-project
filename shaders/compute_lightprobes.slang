#language slang 2026
#include "utils/sdf.slang"
#include "utils/sky.slang"
#include "utils/pbr.slang"



[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id: SV_DispatchThreadID) {   
    
    // [0..TEXTURE_SIZE]
    float3 p = (id.xyz);

    // [-TEXTURE_SIZE/2..TEXTURE_SIZE/2]
    p -= GI_TEXTURE_SIZE/2;

    // [-TEXTURE_SIZE/2..TEXTURE_SIZE/2]
    p *= GI_TEXTURE_SCALING;

    uniforms = load_per_frame_uniforms();

    float3 colour = storage_images_3d[SDF_TEXTURE3_STORAGE_IMAGE_IDX].Load(id.xyz).xyz;

    float2 da = d(p);

    if (da.x > 0.5 || da.x < -0.5) {
        return;
    }

    DifferentialPair<float3> dp_a = diffPair(p);
    bwd_diff(d)(dp_a, 1.0);
    float3 normal = normalize(dp_a.d); 

    float normal_bias = 2.0;
    float rand_strength = 0.9;
    float3 offsetted_point = p + normal * normal_bias;
    float3 new_colour = 0.0;

    for (int i = 0; i < 8; i++) {
        
        float3 n = normalize((hash33(p * 12.432 + uniforms.time + i) - 0.5) * rand_strength + normal);

        TraceRes res = trace(offsetted_point, n);



        /*
        float3 diffuse = get_diffuse_colour_only(offsetted_point, normal);
        PbrSurfaceData surface = PbrSurfaceData(offsetted_point, diffuse, normal, AoRoughnessMetallic(float3(1, 1, 0)));
        CameraData camera = CameraData(res.dir);
        */
        
        float3 sun = uniforms.sun.xyz;
        float3 sun_colour = sun_colour(sun.xyz);
        SunData light = SunData(sun.xyz, 16.0);


        //new_colour = radiance_sun(surface, camera, light);
        
        // we have a hit
        if (res.hit) {
            float3 diffuse = get_diffuse_colour_only(res.pos, res.normal);
            PbrSurfaceData surface = PbrSurfaceData(res.pos, diffuse, res.normal, AoRoughnessMetallic(float3(1, 1, 0)));
            CameraData camera = CameraData(res.dir);

            new_colour += radiance_sun(surface, camera, light) / 8.0;
        } else {
            // float3 reflected = reflect(normalize(offsetted_point - uniforms.position.xyz), normal);
            // new_colour = sample_skybox_and_clouds(res.pos, reflected);
        
            //new_colour = sample_skybox_and_clouds(res.pos, res.dir);
        }

    }


    colour = lerp(colour, new_colour, 0.2);

    storage_images_3d[SDF_TEXTURE3_STORAGE_IMAGE_IDX].Store(id.xyz, float4(colour, 1.0));
}
