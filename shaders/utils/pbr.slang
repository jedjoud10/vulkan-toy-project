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

	// float3 shadow = get_sun_shadow(surface.position, surface.normal, sun.direction); 
	float3 shadow = 1.0;
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

#endif