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

    float2 da = d(p);

    // TODO: should be some factor of the voxel size, accounting for trilinear interpolation
    float distance_threshold = 1.1;

    if (da.x > distance_threshold || da.x < -distance_threshold) {
        return;
    }

    DifferentialPair<float3> dp_a = diffPair(p);
    bwd_diff(d)(dp_a, 1.0);
    float3 normal = normalize(dp_a.d); 

    // should also be a factor of voxel size, so that we offset the point by a full voxel-size unit
    float normal_bias = 0.5;
    
    float rand_strength = 0.9;
    float3 offsetted_point = p + normal * normal_bias;
    float3 new_colour = 0.0;

    int samples = 32;
    for (int i = 0; i < samples; i++) {
        
        float3 n = normalize((hash33(p * 12.432 + i * 100.043253) - 0.5) * 2.0 * rand_strength + normal);

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
            float3 diffuse = get_diffuse_colour_only(res.pos, res.normal, res.data.material);
            PbrSurfaceData surface = PbrSurfaceData(res.pos, diffuse, res.normal, AoRoughnessMetallic(float3(1, 1, 0)));
            CameraData camera = CameraData(-res.dir);

            float3 radiance = radiance_sun(surface, camera, light);
            new_colour += radiance;
        } else {
            new_colour += sample_skybox_and_clouds(res.pos, res.dir);
        }

    }
    
    new_colour /= samples;
    storage_images_3d[SDF_TEXTURE3_STORAGE_IMAGE_IDX].Store(id.xyz, float4(new_colour, 1.0));
}



[shader("compute")]
[numthreads(4, 4, 4)]
void main2(uint3 id: SV_DispatchThreadID) {   
    
    // [0..TEXTURE_SIZE]
    float3 p = (id.xyz);

    // [-TEXTURE_SIZE/2..TEXTURE_SIZE/2]
    p -= GI_TEXTURE_SIZE/2;

    // [-TEXTURE_SIZE/2..TEXTURE_SIZE/2]
    p *= GI_TEXTURE_SCALING;

    uniforms = load_per_frame_uniforms();

    float2 da = d(p);

    // TODO: should be some factor of the voxel size, accounting for trilinear interpolation
    float distance_threshold = 1.1;

    if (da.x > distance_threshold || da.x < -distance_threshold) {
        return;
    }

    DifferentialPair<float3> dp_a = diffPair(p);
    bwd_diff(d)(dp_a, 1.0);
    float3 normal = normalize(dp_a.d); 

    // should also be a factor of voxel size, so that we offset the point by a full voxel-size unit
    float normal_bias = 0.1;
    
    float rand_strength = 0.0;
    float3 offsetted_point = p + normal * normal_bias;
    float3 new_colour = 0.0;


    float3 view = -normalize(uniforms.position.xyz - offsetted_point);
    float3 ray_dir = reflect(view, normal);

    int samples = 64;
    for (int i = 0; i < samples; i++) {
        float3 n = normalize((hash33(p * 12.432 + i * 100.043253) - 0.5) * 2.0 * rand_strength + ray_dir);
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
            float3 diffuse = get_diffuse_colour_only(res.pos, res.normal, res.data.material);
            PbrSurfaceData surface = PbrSurfaceData(res.pos, diffuse, res.normal, AoRoughnessMetallic(float3(1, 1, 0)));
            CameraData camera = CameraData(-res.dir);

            float3 radiance = radiance_sun(surface, camera, light);
            new_colour += radiance;
        } else {
            new_colour += sample_skybox_and_clouds(res.pos, res.dir);
        }

    }
    
    new_colour /= samples;
    storage_images_3d[SDF_TEXTURE4_STORAGE_IMAGE_IDX].Store(id.xyz, float4(new_colour, 1.0));
}