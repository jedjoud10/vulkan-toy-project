#ifndef VXGI
#define VXGI

#include "bindless.slang"
#include "sdf.slang"


public static const float VXGI_TEXTURE_SCALING = 1/4.0;

// using during the rendering stage
public static const float NORMAL_OFFSET = 0.5;
public static const float MIP_COUNT_LIMIT = 5;
public static const float MIN_OPACITY_BREAK = 1.25;
public static const float VXGI_RAY_LENGTH = 16.0;
public static const uint STEP_COUNT = 32;
public static const float DIFFUSE_AMBIENT_HALF_ANGLE_RADIAN = 0.5;

public float4 sample_vxgi(float4 position_lod) {
	return sampled_images_3d[VXGI_SAMPLED_IMAGE_IDX].SampleLevel(samplers[LINEAR_CLAMPED_SAMPLER_IDX], ((position_lod.xyz / VXGI_TEXTURE_SCALING) / VXGI_TEXTURE_SIZE) + 0.5, position_lod.w);
}

// https://github.com/jedjoud10/URP-Testing/blob/main/Assets/VoxelGlobalIllumination.compute#L161-L206
public float3 trace_vxgi(float3 position, float3 normal, float half_angle_radian) {
    float3 ray_pos = position + normal * NORMAL_OFFSET;
    
	float3 colour = 0.0;
    float opacity = 0.0;

    float step_size = VXGI_RAY_LENGTH / STEP_COUNT;
    float3 step = normal * step_size;
    float dist = NORMAL_OFFSET;
    float inv_log_2 = 1.0 / 0.301029;
    float tanned = tan(clamp(half_angle_radian, 0, 3.141 * 0.5));

    for (int i = 0; i < STEP_COUNT; i++) {
        // uses the distance and the half angle to approximate sphere radius at specific distance along cone
        float sphere_radius = dist * tanned;

        // compute mip based on sphere radius. if sphere radius is 0.5, this should result in a mip of 0, since radius would span over an entire voxel
        // FIXME: doesn't account for voxel size frick
        float mip_level = clamp(log(sphere_radius + 1) * inv_log_2, 0, MIP_COUNT_LIMIT);

		// sample from specific mip
		float4 pos = float4(ray_pos, mip_level);

		float4 sampled = sample_vxgi(pos);

		// https://kth.diva-portal.org/smash/get/diva2:1886204/FULLTEXT01.pdf
        colour += (1 - opacity) * sampled.w * sampled.xyz;
		opacity += (1 - opacity) * sampled.w;

		// exit if the sampled voxel is too opaque (meaning that there would be no point in sampling after this voxel; everything would be occluded)
		if (sampled.w > MIN_OPACITY_BREAK) {
			break;
		}

		dist += step_size;
        ray_pos += step;
    }

	return colour;
}

// used during the voxelization stage
public static const float MAX_DISTANCE_THESHOLD = 1.5;
public static const float INSIDE_DISTANCE_OPACITY_MULTIPLIER = 8;
public static const float RADIANCE_MULTIPLIER = 4;

// does this need to be equal to NORMAL_OFFSET?
public static const float OUTSETTING = 0.5;

#ifdef FINITE_DIFF_NORMALS
public float3 get_vxgi_sdf_normal(float3 p) {
    float eps = 0.01;
    float x = d(p - float3(eps, 0, 0)).x;
    float y = d(p - float3(0, eps, 0)).x;
    float z = d(p - float3(0, 0, eps)).x;
    float b = d(p).x;
    return normalize(float3(b-x, b-y, b-z) / eps);
}
#else
public float3 get_vxgi_sdf_normal(float3 p) {
    DifferentialPair<float3> dp_a = diffPair(p);
    bwd_diff(d)(dp_a, 1.0);
    return normalize(dp_a.d); 
}
#endif


#endif