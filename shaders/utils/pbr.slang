#ifndef PBR
#define PBR

#include "sky.slang"
#include "bindless.slang"
#include "sdf.slang"
#include "triplanar.slang"
#include "vxgi.slang"

#define PI 3.1415926538


// Literally the whole implementation is stolen from
// https://www.youtube.com/watch?v=RRE-F57fbXw&ab_channel=VictorGordan
// and https://learnopengl.com/PBR/Lighting

// Normal distribution function
// GGX/Trowbridge-reitz model
float ndf(float roughness, float3 n, float3 h) {
	float a = roughness * roughness;
	float a2 = a * a;

	float n_dot_h = max(dot(n, h), 0.0);	
	float n_dot_h2 = n_dot_h * n_dot_h;	

	float semi_denom = n_dot_h2 * (a2 - 1.0) + 1.0;
	float denom = PI * semi_denom * semi_denom;
	return a2 / denom;
}

// Schlick/GGX model
float g1(float k, float3 n, float3 x) {
	float num = max(dot(n, x), 0);
	float denom = num * (1 - k) + k;
	return num / denom;
}

// Smith model
float gsf(float roughness, float3 n, float3 v, float3 l) {
	float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
	return g1(k, n, v) * g1(k, n, l);
}

// Fresnel function
float3 fresnel(float3 f0, float3 h, float3 v) {
	float cosTheta = clamp(dot(h, v), 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// Fresnel function with roughness
float3 fresnelRoughness(float3 f0, float3 v, float3 x, float roughness) {
	float cosTheta = clamp(1.0 - max(dot(v, x), 0), 0, 1);
	return f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(cosTheta, 5.0);
}

// Cook-torrence model for specular
float3 specular(float3 f0, float roughness, float3 v, float3 l, float3 n, float3 h) {
	float3 num = ndf(roughness, n, h) * gsf(roughness, n, v, l) * fresnel(f0, h, v);
	float denom = 4 * max(dot(v, n), 0.0) * max(dot(l, n), 0.0) + 0.001;
	return num / denom;
}

public struct SunData {
	public float3 direction;
	public float3 color;
};

public struct CameraData {
	public float3 view;
};

public struct AoRoughnessMetallic {
	public float3 inner;

	property float ao {
        get { return inner.x; }
		set { inner.x = newValue; }
	}

	property float roughness {
        get { return inner.y; }
		set { inner.y = newValue; }
	}

	property float metallic {
        get { return inner.z; }
		set { inner.z = newValue; }
	}
}

public struct PbrSurfaceData {
	public float3 position;
	public float3 albedo;
	public float3 normal;
	public AoRoughnessMetallic arm;
};

// https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
float3 GetRandomSphereDir() {
    float2 u = (NextRandomU32() >> uint2(0, 16) & 65535) / 65536.0;

    float phi = u.x * 6.283185307179586;
    float y = u.y * 2 - 1;
    float r = sqrt(1.0 - y * y);
    return float3(sin(phi) * r, y, cos(phi) * r);
}

// PCG random
static uint g_RandomSeed = 0;
uint NextRandomU32() {
    uint state = g_RandomSeed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return g_RandomSeed = (word >> 22u) ^ word;
}
void InitRandom(uint2 pos, uint frameNo) {
    g_RandomSeed = pos.x ^ (pos.y << 16);
    NextRandomU32();
    g_RandomSeed ^= frameNo * 1234;
}

float3 get_sun_shadow(float3 world, float3 normal, float3 sun) {
	float3 randomized_sun_direction = normalize(sun + GetRandomSphereDir() * 0.02);
	let clouds_shadow = 1-clouds(randomized_sun_direction, world).a;

	if (clouds_shadow == 0) {
		return 0.0;
	}

#ifdef SDF_SOFT_SHADOWS_SUPPORTED
	let res = shadow_trace(world, sun);
#else
	let res = shadow_trace(world + normal * THRESHOLD_DISTANCE + randomized_sun_direction * THRESHOLD_DISTANCE, randomized_sun_direction);
#endif	
	
	return min(res.shadow, clouds_shadow);
}


public float3 radiance_sun(
	PbrSurfaceData surface,
	CameraData camera,
	SunData sun,
) {
    float3 half_view = normalize(camera.view + sun.direction);
    float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
	float3 ks = fresnelRoughness(f0, half_view, camera.view, surface.arm.roughness);
	float3 kd = (1 - ks) * (1 - surface.arm.metallic);
	float3 specular = specular(f0, surface.arm.roughness, camera.view, sun.direction, surface.normal, half_view);
	float n_dot_l = max(dot(sun.direction, surface.normal), 0.0);

	float3 shadow = get_sun_shadow(surface.position, surface.normal, sun.direction); 
	return (kd * surface.albedo / PI + specular) * sun.color * n_dot_l * shadow;
}

public float3 lighting2(
	PbrSurfaceData surface,
	CameraData camera,
	SunData light,
) {
	float3 radiance = radiance_sun(surface, camera, light);

	// IBL diffuse term
	float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
	float3 ks = fresnelRoughness(f0, surface.normal, camera.view, surface.arm.roughness);
	float3 kd = (1 - ks) * (1 - surface.arm.metallic);
	float3 irradiance = get_environment_diffuse(surface.normal);
	float3 ambient = irradiance * surface.albedo * kd * surface.arm.ao;
	
	
	return radiance + ambient * 0.1;
}

float3 goo_material(float3 world) {
	return hash33(floor(world/10));
}

float3 get_diffuse_colour_only(float3 world, float3 normal, float material) {
	float3 diffuse = 1.0;

	if ((uniforms.toggles_bitmask & 4) == 0) {
		// TODO: file a bug report SOMEWHERE because these NonUniformResourceIndex are not doing anything
		// (even tried with a simpler setup that uses NonUniformResourceIndex into a texture sample, still does not work) 
		uint material_index = NonUniformResourceIndex(floor((snoise(world.xz * 0.03) * 0.5 + 0.5) * 4));
		uint index = NonUniformResourceIndex(storage_buffers_readonly[STORAGE_BUFFER_MATERIALS_BUFFER].Load<uint32_t>(NonUniformResourceIndex(material_index * 4)));
	
	
		diffuse = sample_triplanar(index, world, normal);

		diffuse = lerp(diffuse, goo_material(world), material);
	}

	return diffuse;
}



public float3 trace_scene_multibounce(uint2 id, float3 ray_pos, float3 ray_dir) {
	float3 col = 0.0;
	float3 mask = 1.0;
	InitRandom(id.xy, (uint32_t)uniforms.frame_count);

	float3 camera_view = -ray_dir;

	// https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
	for (int bounce = 0; bounce < 2; bounce++) {
		TraceRes res = trace(ray_pos, ray_dir);

    	if (res.hit) {  
			float3 world = res.pos;
			float3 normal = res.normal;

			float3 diffuse = 1.0; 
			float3 arm = float3(1, 1, 0);

			if (res.data.instance_custom_index != 0) {
				arm = float3(1, 0.8, 0);

				// tree colours
				float3 trunk = hex2col(0x564631) * (noise(world * 20.2) * 0.2 + 0.9);
				float3 leaf = hex2col(0x588732) * (noise(world * 10.0) * 0.2 + 0.9);
				diffuse = lerp(trunk, leaf, res.data.material);
				

				// cheese colours
				//diffuse = hex2col(0xf7de51) * (noise(world * 10.0) * 0.3 + 0.5);
			}

			if ((uniforms.toggles_bitmask & 4) == 0 && res.data.instance_custom_index == 0) {
				// TODO: file a bug report SOMEWHERE because these NonUniformResourceIndex are not doing anything
				// (even tried with a simpler setup that uses NonUniformResourceIndex into a texture sample, still does not work) 
				uint material_index = NonUniformResourceIndex(floor((snoise(world.xz * 0.03) * 0.5 + 0.5) * 4));
				uint index = NonUniformResourceIndex(storage_buffers_readonly[STORAGE_BUFFER_MATERIALS_BUFFER].Load<uint32_t>(NonUniformResourceIndex(material_index * 4)));
			

				diffuse = sample_triplanar(index, world, normal);
				arm = sample_triplanar(index+1, world, normal);
				normal = sample_triplanar_normal(index+2, world, normal);
			
				diffuse = lerp(diffuse, 1.0, res.data.material);
				arm = lerp(arm, float3(1, 0.0, 1.0), res.data.material);
				normal = normalize(lerp(normal, res.normal, res.data.material));
			}



			
			PbrSurfaceData surface = PbrSurfaceData(res.pos, diffuse, normal, AoRoughnessMetallic(arm));
			CameraData camera = CameraData(camera_view);
			
			// opaque surface, non refractive
			float3 sun = uniforms.sun.xyz;
			float3 sun_colour = sun_colour(sun.xyz);
			SunData light = SunData(sun.xyz, 16.0);

			float3 radiance = radiance_sun(surface, camera, light);

			// IBL diffuse term
			float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
			float3 ks = fresnelRoughness(f0, surface.normal, camera.view, surface.arm.roughness);
			float3 kd = (1 - ks) * (1 - surface.arm.metallic);
			
			// TODO: there's still a problem here. plasticy materials (roughness~0, metallic~0) look like metals. I'm forgetting to account for something here. I don't know my PBR
			col += mask * radiance;
			//float3 ambient = surface.albedo * kd * surface.arm.ao;
			mask *= lerp(fresnelRoughness(surface.albedo, surface.normal, camera.view, surface.arm.roughness), surface.albedo, surface.arm.metallic);

			float3 random_normal = normal + GetRandomSphereDir();
			float3 reflected_ray_dir = reflect(ray_dir, normal);

			// TODO: this is also wrong. instead of shooting *one* ray for the surface we need to shoot the ambient and reflection rays separately. we cannot just shoot one to approximate both effect
			ray_dir = normalize(lerp(reflected_ray_dir, random_normal, surface.arm.roughness));
			ray_pos = res.pos + ray_dir * 0.01;
			camera_view = -ray_dir;
    	} else {
    	    col += mask * sample_skybox_and_clouds(res.pos, res.dir);
			break;
    	}
	}


	return col;
}

public float3 trace_scene_singlebounce(uint2 id, float3 ray_pos, float3 ray_dir) {
	float3 col = 0.0;

	float3 camera_view = -ray_dir;

	TraceRes res = trace(ray_pos, ray_dir);
	InitRandom(id.xy, (uint32_t)uniforms.frame_count);

	if (res.hit) {  
		float3 world = res.pos;
		float3 normal = res.normal;
		float3 geometry_normal = res.normal;

		float3 diffuse = 1.0; 
		float3 arm = float3(1, 1, 0);

		if (res.data.instance_custom_index != 0) {
			arm = float3(1, 0.8, 0);

			// tree colours
			float3 trunk = hex2col(0x564631) * (noise(world * 20.2) * 0.2 + 0.9);
			float3 leaf = hex2col(0x588732) * (noise(world * 10.0) * 0.2 + 0.9);
			diffuse = lerp(trunk, leaf, res.data.material);
			

			// cheese colours
			//diffuse = hex2col(0xf7de51) * (noise(world * 10.0) * 0.3 + 0.5);
		}

		if ((uniforms.toggles_bitmask & 4) == 0 && res.data.instance_custom_index == 0) {
			// TODO: file a bug report SOMEWHERE because these NonUniformResourceIndex are not doing anything
			// (even tried with a simpler setup that uses NonUniformResourceIndex into a texture sample, still does not work) 
			uint material_index = NonUniformResourceIndex(floor((snoise(world.xz * 0.03) * 0.5 + 0.5) * 4));
			uint index = NonUniformResourceIndex(storage_buffers_readonly[STORAGE_BUFFER_MATERIALS_BUFFER].Load<uint32_t>(NonUniformResourceIndex(material_index * 4)));
		

			diffuse = sample_triplanar(index, world, normal);
			arm = sample_triplanar(index+1, world, normal);
			normal = sample_triplanar_normal(index+2, world, normal);
		
			diffuse = lerp(diffuse, goo_material(world), res.data.material);
			arm = lerp(arm, float3(1, 1.0, 0.0), res.data.material);
			normal = normalize(lerp(normal, res.normal, res.data.material));
		}
		
		PbrSurfaceData surface = PbrSurfaceData(res.pos, diffuse, normal, AoRoughnessMetallic(arm));
		CameraData camera = CameraData(camera_view);
		
		// opaque surface, non refractive
		float3 sun = uniforms.sun.xyz;
		float3 sun_colour = sun_colour(sun.xyz);
		SunData light = SunData(sun.xyz, 16.0);

		float3 radiance = radiance_sun(surface, camera, light);

		// IBL diffuse term
		float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
		float3 ks = fresnelRoughness(f0, surface.normal, camera.view, surface.arm.roughness);
		float3 kd = (1 - ks) * (1 - surface.arm.metallic);

		float3 sampled_ambient = trace_vxgi(world, geometry_normal, DIFFUSE_AMBIENT_HALF_ANGLE_RADIAN);
		/*
		// this is "more correct", to approximate diffuse lighting, but surprisingly, shooting only one cone works fine for now
		float3 sampled_ambient = 0.0;
		for (int r = 0; r < 8; r++) {
			sampled_ambient += trace_vxgi(world, normalize(normal + ((hash33(r * 100.5342 + world * 54.23423) - 0.5) * 1.0)), DIFFUSE_AMBIENT_HALF_ANGLE_RADIAN) / 8.0;
		}
		*/
		sampled_ambient += get_environment_diffuse(geometry_normal) * 0.2;
		float3 ambient = surface.albedo * kd * surface.arm.ao * sampled_ambient;
		col += radiance + ambient;

		// HACK: hacky way to get reflections
		// ideally we would figure out if a ray is "mirror like", and instead trace a new ray from scratch instead of using the VXGI map 
		// horribly slow on the mud material, probably because of high thread divergence and chache line misses
		/*
		if (surface.arm.roughness < 1.0) {
			float3 reflected = reflect(ray_dir, normal);
			col += trace_vxgi(world, reflected, surface.arm.roughness * 0.1) * ks;
			col += sample_skybox_and_clouds(world, reflected) * ks * (1 - surface.arm.roughness);
		}
		*/

		if ((uniforms.toggles_bitmask & 8) != 0) {
			col = ambient;
		}

		if ((uniforms.toggles_bitmask & 16) != 0) {
			float4 k = sample_vxgi(float4(world - res.normal * 0.3, 0));
			col = k.xyz;
		}
		
	} else {
		col = sample_skybox_and_clouds(res.pos, res.dir);
	}


	return col;
}

#endif