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


[SpecializationConstant]
const bool SUN_SHADOW_RAY = false;

[SpecializationConstant]
const bool EXTRA_LIGHTS_ENABLED = false;

[SpecializationConstant]
const bool EXTRA_LIGHTS_SHADOW_RAY = false;

[SpecializationConstant]
const bool VERY_SHINY_REFLECTIVE_SURFACES = false;

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

	    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE> sq;
	    let flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE;

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

    	RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE> sq;
    	let flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE;

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
	bool clouds_not_shadowed = shadow > 0.0;
	bool face_visible = dot(normal, sun_direction) > 0;
	if (clouds_not_shadowed && face_visible) {
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
    float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
	float3 ks = fresnelRoughness(f0, half_view, camera.view, surface.arm.roughness);
	float3 kd = (1 - ks) * (1 - surface.arm.metallic);
	float3 specular = specular(f0, surface.arm.roughness, camera.view, sun.direction, surface.normal, half_view);
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
    float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
	float3 ks = fresnelRoughness(f0, half_view, camera.view, surface.arm.roughness);
	float3 kd = (1 - ks) * (1 - surface.arm.metallic);
	float3 specular = specular(f0, surface.arm.roughness, camera.view, light_direction, surface.normal, half_view);
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
	float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
	float3 ks = fresnelRoughness(f0, surface.normal, camera.view, surface.arm.roughness);
	float3 kd = (1 - ks) * (1 - surface.arm.metallic);
	float3 irradiance = get_environment_diffuse(surface.normal);
	float3 ambient = irradiance * surface.albedo * kd * surface.arm.ao;
	
	
	return radiance + ambient * 0.1;
}


public float3 lighting3<let K: uint>(
	PbrSurfaceData surface,
	CameraData camera,
	SunData light,
) {
	float3 radiance = radiance_sun(surface, camera, light);
	
	if (EXTRA_LIGHTS_ENABLED) {
		for (int i = 0; i < EXTRA_LIGHTS_NUM; i++) {
			float4 light = storage_buffers_readonly[STORAGE_BUFFER_LIGHTS_BUFFER].Load<float4>(i * 4*4);
			radiance += radiance_light(surface, camera, light) * 20;
		}
	}
	
	// IBL diffuse term
	float3 f0 = lerp(0.04, surface.albedo, surface.arm.metallic);
	float3 ks = fresnelRoughness(f0, surface.normal, camera.view, surface.arm.roughness);
	float3 kd = (1 - ks) * (1 - surface.arm.metallic);
	float3 irradiance = get_environment_diffuse(surface.normal);
	float3 ambient = irradiance * surface.albedo * kd * surface.arm.ao;

	// shitty IBL specular term using RT
	// does not work well for high roughness because we'd need to permute and randomize the rays for the roughness look
	// too expensive, need to figure out a nice approx
	let uniforms = load_per_frame_uniforms();
	if (VERY_SHINY_REFLECTIVE_SURFACES && (uniforms.toggles_bitmask & 4) == 4) {
		float3 ray_pos = surface.position;
		int samples = 1;
		for (int l = 0; l < samples; l++) {
			float3 perturb = (hash33(surface.position * float3(34.2143, -50.54353, 90.2342) + l * 400.523423 + uniforms.time) * 2.0 - 1.0) * surface.arm.roughness;
			//float3 perturb = 0;
			float3 ray_dir = reflect(-camera.view, normalize(surface.normal + perturb));
			if (surface.arm.roughness < 0.9) {
				if (K > 0) {
					RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> sq;
    				let flags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

					RayDesc desc;		

					desc.Origin = ray_pos;
					desc.Direction = ray_dir;
					desc.TMax = 100;
					desc.TMin = 0.001;

					sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);

					while (sq.Proceed()) {
						if(sq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
							sq.CommitNonOpaqueTriangleHit();  // forcing to be opaque
					}
				
					if (sq.CommittedStatus() == COMMITTED_NOTHING) {
						radiance += sample_skybox_and_clouds(ray_pos, ray_dir) * ks * 1.0 / (float)samples;
					} else {
						int custom_instance_index = sq.CommittedRayInstanceCustomIndex();
						
						uint instance_index = sq.CommittedInstanceIndex();
						GpuModelMetadata metadata = storage_buffers_readonly[STORAGE_BUFFER_MODELS_METADATA_IDX].Load<GpuModelMetadata>(instance_index * sizeof(GpuModelMetadata));

						let index_buffer = storage_buffers_readonly[NonUniformResourceIndex(metadata.storage_buffers_base_index)];
						let vertex_normals = storage_buffers_readonly[NonUniformResourceIndex(metadata.storage_buffers_base_index+1)];
						let vertex_uvs = storage_buffers_readonly[NonUniformResourceIndex(metadata.storage_buffers_base_index+2)];


						uint triangle_index = sq.CommittedPrimitiveIndex();
						uint3 indices = index_buffer.Load<uint3>(triangle_index * sizeof(uint3)); 

						float2 barycentrics = sq.CommittedRayBarycentrics();

						float3 n0 = vertex_normals.Load<float3>(indices.x * sizeof(float3));
						float3 n1 = vertex_normals.Load<float3>(indices.y * sizeof(float3));
						float3 n2 = vertex_normals.Load<float3>(indices.z * sizeof(float3));

						float2 uv0 = vertex_uvs.Load<float2>(indices.x * sizeof(float2));
						float2 uv1 = vertex_uvs.Load<float2>(indices.y * sizeof(float2));
						float2 uv2 = vertex_uvs.Load<float2>(indices.z * sizeof(float2));




						float3 world = ray_pos + ray_dir * sq.CommittedRayT();
						float3[3] tris = sq.CommittedGetIntersectionTriangleVertexPositions();

    					// I eventually need to understand why this works but I don't...
    					float3 dp1 = tris[1] - tris[0];
    					float3 dp2 = tris[2] - tris[0];
    					float2 duv1 = uv1 - uv0;
    					float2 duv2 = uv2 - uv0;

    					// solve for tangent and bitangent
    					float3 normal = normalize(cross(dp1, dp2));
    					float3 tangent = normalize(duv2.y * dp1 - duv1.y * dp2);
    					float3 bitangent = normalize(-duv2.x * dp1 + duv1.x * dp2);
    					float3x3 tbn = float3x3(tangent, bitangent, normal);

    					if ((uniforms.toggles_bitmask & 1) == 1) {
    					    normal = n0 * (1 - (barycentrics.x + barycentrics.y)) + n1 * barycentrics.x + n2 * barycentrics.y;
							normal = mul(sq.CommittedObjectToWorld3x4(), float4(normal, 0.0)).xyz;
    					}

						float2 uv = uv0 * (1 - (barycentrics.x + barycentrics.y)) + uv1 * barycentrics.x + uv2 * barycentrics.y;
					
    					// sample textures
    					let sampler = samplers[NEAREST_SAMPLER_IDX];
    					float3 diffuse = sampled_images[NonUniformResourceIndex(metadata.material_base_index)].Sample(sampler, uv).xyz;
    					float3 arm = sampled_images[NonUniformResourceIndex(metadata.material_base_index+1)].Sample(sampler, uv).xyz;
					
    					// calculate world space normal perturbed by sampled tangent normal map
    					if ((uniforms.toggles_bitmask & 2) == 2) {
    					    float3 tangent_normal = sampled_images[NonUniformResourceIndex(metadata.material_base_index+2)].Sample(sampler, uv).xyz * 2.0 - 1.0;
    					    normal = mul(tangent_normal, tbn); 
							normal = mul(sq.CommittedObjectToWorld3x4(), float4(normal, 0.0)).xyz;
    					}


						surface.position = world;
						camera.view = -ray_dir;


						surface.albedo = diffuse;
						surface.normal = normalize(normal);
						surface.arm.inner = arm;
						//surface.arm.inner = float3(1, 1, 0);

						//radiance += float3(uv%1.0, 0);
						radiance += lighting3<K-1>(surface, camera, light) * ks * (1.0/(float)samples);
					}
				} else {
					// this fallback is shit... need to implement specular IBL for varying roughnesses
					radiance += sample_skybox_and_clouds(ray_pos, ray_dir) * ks * (1.0/(float)samples);
				}
			}
		}
		
	}
	
	
	
	
	return radiance + ambient * 0.3;
}

#endif