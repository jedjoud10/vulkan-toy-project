#language slang 2026

#include "utils/noises.slang"
#include "utils/sky.slang"
#include "utils/bindless.slang"


[SpecializationConstant] const uint SKYBOX_RESOLUTION = 32;
[SpecializationConstant] const uint CLOUDS_RESOLUTION = 32;
[SpecializationConstant] const uint AMBIENT_SKYBOX_RESOLUTION = 32;

[shader("compute")]
[numthreads(8, 8, 1)]
void write_clouds(uint3 id: SV_DispatchThreadID) {
    float2 uv = id.xy / (float)CLOUDS_RESOLUTION;
    
    let uniform_thing = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float3 sun = uniform_thing.sun.xyz;
    float time = uniform_thing.time;

    float2 pos = (uv-0.5) * 8000;

    float amplitude = 0.7;
    float scale = 2.3;
    float3 value = 0;

    for (int i = 0; i < 4; i++) {
        let noised = sdnoise(pos * pow(scale, i) * 0.0015 + time * 0.03) * pow(amplitude, i);
        value += noised;
    }

    // scale opacity by another noisy function
    float opacity = value.x * smoothstep(0, 1, snoise(pos * 0.0005) * 1.5 - 0.2) * 6.0;
    float2 derivatives = value.yz;

    //opacity *= saturate(3000-length(pos));

    float3 ray_dir = normalize(float3(pos.x, 800, pos.y));

    float3 bottom_normals = normalize(float3(derivatives.x, -1, derivatives.y));
    float3 top_normals = normalize(float3(-derivatives.x, 1, -derivatives.y));

    // factor to use to "grey out" the clouds when the sun is down
    float sun_strength = smoothstep(0, 0.2, sun.y);

    // backlight from the sun that passes through the clouds
    float3 scattered = saturate(pow(saturate(dot(ray_dir, sun)) + 0.1, 16)) * sun_strength;

    // some specular "silver lining"
    float3 reflected = reflect(sun, bottom_normals);
    float3 silver_lining = pow(saturate(dot(ray_dir, reflected)), 0.5) * sun_strength;

    float3 ambient = sky(sun, top_normals, 0, false, false);

    // final composition
    float3 col = (lerp(silver_lining * 0.3, 1.4, scattered) + 0.4) * (ambient * 1.0 + 0.3);
    (storage_images[CLOUDS_STORAGE_IMAGE_IDX])[id.xy] = float4(col, saturate(opacity));
}

float3 uv_coords_to_ray_dir(float2 uv, int cubemap_face) {
    float3 coords = 0;

    // -X
    if (cubemap_face == 0) {
        coords = float3(-1, -uv.y, uv.x); 
    }

    // +X
    if (cubemap_face == 1) {
        coords = float3(1, -uv.y, -uv.x);
    }

    // -Y
    if (cubemap_face == 3) {
        coords = float3(-uv.x, -1, uv.y);
    }

    // +Y
    if (cubemap_face == 2) {
        coords = float3(-uv.x, 1, -uv.y);
    }

    // -Z
    if (cubemap_face == 4) {
        coords = float3(-uv.x, -uv.y, -1);
    }

    // +Z
    if (cubemap_face == 5) {
        coords = float3(uv.x, -uv.y, 1);
    }

    float3 ray_dir = normalize(coords);
    ray_dir.xz = -ray_dir.xz;
    return ray_dir;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void write_skybox(uint3 id: SV_DispatchThreadID, uint3 gid: SV_GroupID) {
    float2 uv = ((id.xy / (float)SKYBOX_RESOLUTION) * 2.0 - 1.0);

    let uniform_thing = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float3 sun = uniform_thing.sun.xyz;
    float time = uniform_thing.time;

    
    float3 ray_dir = uv_coords_to_ray_dir(uv, gid.z);
    float3 sky_colour = sky(sun, ray_dir, 0.0, false, true);
    (storage_images_array[SKYBOX_STORAGE_IMAGE_IDX]).Store(id, float4(sky_colour, 0));
}

groupshared Atomic<float> color_r;
groupshared Atomic<float> color_g;
groupshared Atomic<float> color_b;

[shader("compute")]
[numthreads(32, 16, 1)]
void blur_skybox_ambient(uint3 id: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    if (all(gtid == 0)) {
        color_r = 0;
        color_g = 0;
        color_b = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    float2 uv = ((gid.xy / (float)AMBIENT_SKYBOX_RESOLUTION) * 2.0 - 1.0);

    let uniform_thing = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    float3 sun = uniform_thing.sun.xyz;
    float time = uniform_thing.time;

    float3 ray_dir = uv_coords_to_ray_dir(uv, gid.z);

    float3 irradiance = float3(0.0);
    float3 up = float3(0.0, 1.0, 0.0);
    float3 right = normalize(cross(up, ray_dir));
    up = normalize(cross(ray_dir, right));

    // Fix that one pixel on the Y+ cubemap plane and Y- cubemap plane
    if (abs(ray_dir.y) == 1) {
        up = vec3(0, 0, 1);
        right = vec3(1, 0, 0);
    }

    const float PI = 3.14159265359;
    float sample_delta = 0.125;
    float samples = 0.0;

    const float ANGLE_PHI_STEPS = 32.0;
    const float ANGLE_THETA_STEPS = 32.0;
    

    // https://learnopengl.com/PBR/IBL/Diffuse-irradiance
    float phi = (gtid.x / ANGLE_PHI_STEPS) * 2.0 * PI;
    float theta = (gtid.y / ANGLE_THETA_STEPS) * 0.5 * PI;
    float3 sampled_tangent = float3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
    float3 sampled_dir = sampled_tangent.x * right + sampled_tangent.y * up + sampled_tangent.z * ray_dir; 

    float3 sampled = sample_skybox_and_clouds(0, normalize(sampled_dir));
    color_r.add(sampled.x, MemoryOrder.Relaxed);
    color_g.add(sampled.y, MemoryOrder.Relaxed);
    color_b.add(sampled.z, MemoryOrder.Relaxed);

    GroupMemoryBarrierWithGroupSync();

    // Return the irradiance diffuse lighting
    float3 color = PI * (float3(color_r.load(MemoryOrder.Relaxed), color_g.load(MemoryOrder.Relaxed), color_b.load(MemoryOrder.Relaxed))) / (ANGLE_PHI_STEPS*ANGLE_THETA_STEPS);
    (storage_images_array[AMBIENT_SKYBOX_STORAGE_IMAGE_IDX]).Store(gid.xyz, float4(color, 0));

    /*
    // https://learnopengl.com/PBR/IBL/Diffuse-irradiance
    [loop]
    for(float phi = 0.0; phi < 2.0 * PI; phi += sample_delta) {
        [loop]
        for(float theta = 0.0; theta < 0.5 * PI; theta += sample_delta) {
            float3 sampled_tangent = float3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            float3 sampled_dir = sampled_tangent.x * right + sampled_tangent.y * up + sampled_tangent.z * ray_dir; 

            irradiance += sample_skybox_and_clouds(0, normalize(sampled_dir));
            samples += 1.0;
        }
    }

    // Return the irradiance diffuse lighting
    float3 color = PI * irradiance * (1.0 / samples);
    (storage_images_array[AMBIENT_SKYBOX_STORAGE_IMAGE_IDX]).Store(id, float4(color, 0));
    */
}