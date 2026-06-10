#ifndef PBR
#define PBR

#include "sky.slang"
#include "bindless.slang"

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

// Sun data struct
public struct SunData {
	public float3 direction;
	public float3 color;
};
// Camera data struct
public struct CameraData {
	public float3 view;
};

// Surface data struct 
public struct PbrSurfaceData {
	public float3 position;
	public float3 albedo;
	public float3 normal;
	public float3 arm;
};


[SpecializationConstant]
const bool SUN_SHADOW_RAY = false;

[SpecializationConstant]
const bool EXTRA_LIGHTS_ENABLED = false;

[SpecializationConstant]
const bool EXTRA_LIGHTS_SHADOW_RAY = false;

[SpecializationConstant]
const uint EXTRA_LIGHTS_NUM = 1;


float hw_trace_shadow_towards_sun(float3 world, float3 normal, float3 sun_direction) {
	if (SUN_SHADOW_RAY) {
	    static const float SHADOW_RAY_NORMAL_OFFSET = 0.001;
	    static const float SHADOW_RAY_TMAX = 1000; 
	    static const float SHADOW_RAY_TMIN = 0.01; 
	
	    RayDesc desc;
	    desc.Origin = world + normal * SHADOW_RAY_NORMAL_OFFSET;
	    desc.Direction = sun_direction;
	    desc.TMax = SHADOW_RAY_TMAX;
	    desc.TMin = SHADOW_RAY_TMIN;

	    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> sq;
	    let flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE;

	    sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);

	    sq.Proceed();

	    return (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
	} else {
	    return 1.0;
	}
}

float hw_trace_shadow_light(float3 world, float3 light) {
	if (EXTRA_LIGHTS_SHADOW_RAY) {
		RayDesc desc;
    	desc.Origin = world;
    	desc.Direction = normalize(light-world);
    	desc.TMax = distance(world, light);
    	desc.TMin = 0.01;

    	RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> sq;
    	let flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE;

    	sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);

    	sq.Proceed();

    	return (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
	} else {
		return 1.0;
	}
}

float3 get_sun_shadow(float3 world, float3 normal, float3 sun_direction) {
	// TODO: clouds let a bit of light through, but it is scattered. take account cloud colour
	float shadow = 1-clouds(sun_direction, world).a;

	// little trick to avoid tracing the shadow ray in case clouds already cover the sun's light
	if (shadow > 0.0) {
		shadow *= hw_trace_shadow_towards_sun(world, normal, sun_direction);
	}

	return shadow;
}

float3 get_light_shadow(float3 world, float3 light) {
	return hw_trace_shadow_light(world, light);
}


public float3 radiance_sun(
	PbrSurfaceData surface,
	CameraData camera,
	SunData sun,
) {
    float3 half_view = normalize(camera.view + sun.direction);
    float3 f0 = lerp(0.04, surface.albedo, surface.arm.z);
	float3 ks = fresnelRoughness(f0, half_view, camera.view, surface.arm.y);
	float3 kd = (1 - ks) * (1 - surface.arm.z);
	float3 specular = specular(f0, surface.arm.y, camera.view, sun.direction, surface.normal, half_view);
	float n_dot_l = max(dot(sun.direction, surface.normal), 0.0);

	float3 shadow = get_sun_shadow(surface.position, surface.normal, sun.direction); 

	return (kd * surface.albedo / PI + specular) * sun.color * n_dot_l * shadow;
}

public float3 radiance_light(
	PbrSurfaceData surface,
	CameraData camera,
	float4 light,
) {
	float3 light_position = light.xyz;
	float distance = length(light_position - surface.position);
	float attenuation = 1.0 / (distance*distance);
	float3 light_direction = normalize(light_position - surface.position);
	
    float3 half_view = normalize(camera.view + light_direction);
    float3 f0 = lerp(0.04, surface.albedo, surface.arm.z);
	float3 ks = fresnelRoughness(f0, half_view, camera.view, surface.arm.y);
	float3 kd = (1 - ks) * (1 - surface.arm.z);
	float3 specular = specular(f0, surface.arm.y, camera.view, light_direction, surface.normal, half_view);
	float n_dot_l = max(dot(light_direction, surface.normal), 0.0);

	float3 shadow = get_light_shadow(surface.position, light.xyz); 

	return attenuation * (kd * surface.albedo / PI + specular) * n_dot_l * shadow;
}


public float3 lighting2(
	PbrSurfaceData surface,
	CameraData camera,
	SunData light,
) {
	float3 radiance = radiance_sun(surface, camera, light);
	
	if (EXTRA_LIGHTS_ENABLED) {
		for (int i = 0; i < EXTRA_LIGHTS_NUM; i++) {
			float4 light = storage_buffers_readonly[STORAGE_BUFFER_LIGHTS_BUFFER].Load<float4>(i * 4*4);
			radiance += radiance_light(surface, camera, light);
		}
	}
	
	// IBL diffuse term
	float3 f0 = lerp(0.04, surface.albedo, surface.arm.z);
	float3 ks = fresnelRoughness(f0, surface.normal, camera.view, surface.arm.y);
	float3 kd = (1 - ks) * (1 - surface.arm.z);
	float3 irradiance = get_environment_diffuse(surface.normal);
	float3 ambient = irradiance * surface.albedo * kd * surface.arm.x;
	
	
	return radiance + ambient * 0.1;
}

#endif