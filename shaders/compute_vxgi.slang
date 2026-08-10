#include "utils/bindless.slang"
#include "utils/pbr.slang"

static const int TEXTURE_RES = GI_TEXTURE_SIZE;
static const float MAX_DISTANCE_THESHOLD = 1.2;

float3 test(float3 p)
{
    float eps = 0.01;
    float x = d(p - float3(eps, 0, 0)).x;
    float y = d(p - float3(0, eps, 0)).x;
    float z = d(p - float3(0, 0, eps)).x;
    float b = d(p).x;


    return (float3(b-x, b-y, b-z) / eps);
}


[shader("compute")]
[numthreads(4,4,4)]
void voxelize (uint3 id : SV_DispatchThreadID) {
    // [0..TEXTURE_SIZE]
    float3 p = (id.xyz + 1.0);

    // [-TEXTURE_SIZE/2..TEXTURE_SIZE/2]
    p -= GI_TEXTURE_SIZE/2;

    // [-MAP_SIZE/2..MAP_SIZE/2]
    p *= GI_TEXTURE_SCALING;

    uniforms = load_per_frame_uniforms();
    float2 da = d(p);

    if (da.x > MAX_DISTANCE_THESHOLD) {
        return;
    }

    float m = da.y;


    /*
    DifferentialPair<float3> dp_a = diffPair(p);
    bwd_diff(d)(dp_a, 1.0);
    float3 n = normalize(dp_a.d); 
    */

    float3 n = normalize(test(p));

    if (any(isnan(n))) {
        n = 0;
    }

    // offset point by normal a bit
    //p -= n;

    /*
	int count = 0;

	// TODO: optimize this using sparse structure
	// this kernel is taking 2.5ms on my machine. That is nearly 50% of the entire time spent doing VXGI (total of 5.5ms)
	for (int i = 0; i < numBoundsStuff; i++) {
		Bound t = unpack_bound(voxelizationObjectBounds[i]);

		float3 center = (t.max.xyz + t.min.xyz) * 0.5;
		float3 size = (t.max.xyz - t.min.xyz) * 0.5;

		
		// we only use the matrices for rotations since we already know center and size
		float3 rotated = rotate_vector(center - p, q_inverse(t.quaternion));

		// float4 sampled = sdgBox(center - p, size, 0);
		float4 sampled = sdgBox(rotated, size, 0.0);

		// we just need to find the FIRST intersecting OBB
		// we can exit afterwards
		if (sampled.x < threshold) {
			colour += float4(t.emissive.xyz, 1);
			n += rotate_vector(normalize(sampled.yzw), t.quaternion).xyz;
			count += 1;
		}
	}
    
    
	mainTex[id.xyz] = colour / max(count, 1);
	normalsTex[id.xyz] = float4(-normalize(n), 1);
    */

    /*
    if (any(isnan(n))) {
        storage_images_3d[SDF_TEXTURE3_STORAGE_IMAGE_IDX].Store(id, float4(100));
        return;
    }    
    */


    float3 diffuse = get_diffuse_colour_only(p, n, m);
    PbrSurfaceData surface = PbrSurfaceData(p, diffuse, n, AoRoughnessMetallic(float3(1, 1, 0)));
    SunData sun = SunData(uniforms.sun.xyz, 16.0);

    // shouldn't matter since we consider fully diffuse materials
    CameraData camera = CameraData(n);
        
	float3 colour = radiance_sun(surface, camera, sun) * 2.0;

    // colour += sample_skybox_and_clouds(p, n) * 0.5;

    if (any(isnan(colour))) {
        colour = 0;
    }

    storage_images_3d[SDF_TEXTURE3_STORAGE_IMAGE_IDX].Store(id, float4(colour, 1.0));
    //storage_images_3d[SDF_TEXTURE4_STORAGE_IMAGE_IDX].Store(id, float4(n, 1));
    
}

[shader("compute")]
[numthreads(4,4,4)]
void mip_map_downsample(uint3 id : SV_DispatchThreadID, uniform uint src_mip_sampled_image_index, uniform uint dst_storage_mip_index) {
	float4 summed = 0;
	const static int size = 2;
	const static int volume = (size*2 + 1) * (size*2 + 1) * (size*2 + 1); 
	const static float scale = 3.0;

	float max_alpha = 0;

	for (int x = -size; x <= size; x++) {
		for (int y = -size; y <= size; y++) {
			for (int z = -size; z <= size; z++) {
                float3 texture_position = id.xyz * 2 + float3(x,y,z) * scale;

				float4 loaded = sampled_images_3d[src_mip_sampled_image_index].Sample(samplers[LINEAR_CLAMPED_SAMPLER_IDX], texture_position / TEXTURE_RES);
				summed += loaded;
			}
		}	
	}

	storage_images_3d[dst_storage_mip_index].Store(id, summed / volume);	
}

/*
AppendStructuredBuffer<uint> surfaceVoxels;
RWStructuredBuffer<uint4> indirectArgs;
int ambientTextureScalingFactor;
RWTexture3D<float4> ambientTex;

float indirectDiffuseFactor;
float originalDiffuseFactor;

[shader("compute")]
[numthreads(8,8,8)]
void CSGenerateSurfaces(uint3 id : SV_DispatchThreadID) {
	uint3 src_tex_id = id.xyz * ambientTextureScalingFactor;
	
	ambientTex[id.xyz] = mainTex[src_tex_id] * originalDiffuseFactor;
	
	int k = ambientTextureScalingFactor;
	float threshold = 0.1;
	bool origin = mainTex[src_tex_id].w > threshold;

	if (!origin) {
		return;
	}

	bool offset_px = mainTex[src_tex_id + int3(k, 0, 0)].w > threshold;
	bool offset_nx = mainTex[src_tex_id + int3(-k, 0, 0)].w > threshold;

	bool offset_py = mainTex[src_tex_id + int3(0, k, 0)].w > threshold;
	bool offset_ny = mainTex[src_tex_id + int3(0, -k, 0)].w > threshold;
	
	bool offset_pz = mainTex[src_tex_id + int3(0, 0, k)].w > threshold;
	bool offset_nz = mainTex[src_tex_id + int3(0, 0, -k)].w > threshold;

	if (origin && (!offset_px || !offset_nx || !offset_py || !offset_ny || !offset_pz || !offset_nz)) {
		surfaceVoxels.Append(id_to_index(id));
	}
}

[shader("compute")]
[numthreads(1,1,1)]
void CSCreateIndirectDispatchArgs(uint3 id : SV_DispatchThreadID) {
	uint num_surfaces = indirectArgs[0].x;
	uint4 value = 0;
	value.x = ((num_surfaces + 32 - 1) / 32);
	value.y = 1;
	value.z = 1;
	value.w = num_surfaces;
	indirectArgs[0] = value;
}


#ifdef QUALITY_LOW
static const int stepCount = 16;
static const int randDirs = 32;
#endif

#ifdef QUALITY_MID
static const int stepCount = 32;
static const int randDirs = 32;
#endif

#ifdef QUALITY_HIGH
static const int stepCount = 32;
static const int randDirs = 256;
#endif


sampler3D voxelGiTex;
sampler3D upsampledVoxelGiTex;
float normalMult;
float stepSize;
float startOffset;
float diffuseHalfAngle;
float mipMapCountLimit;
float minOpacityBreak;
StructuredBuffer<uint> readOnlySurfaceVoxels;

float3 TraceStuff(float3 position, float3 normal, float half_angle_radian, inout int total) {
    float3 ray_pos = position + normal * startOffset;
    
	float3 colour = 0.0;
    float opacity = 0.0;

    float step_size = stepSize / stepCount;
    float3 step = normal * step_size;
    float dist = startOffset;
    float inv_log_2 = 1.0 / 0.301029;
    float tanned = tan(clamp(half_angle_radian, 0, 3.141 * 0.5));

    for (int i = 0; i < stepCount; i++) {
        dist += step_size;
        ray_pos += step;

        // uses the distance and the half angle to approximate sphere radius at specific distance along cone
        float sphere_radius = dist * tanned;

        // compute mip based on sphere radius. if sphere radius is 0.5, this should result in a mip of 0, since radius would span over an entire voxel
        float mip_level = clamp(log(sphere_radius + 1) * inv_log_2, 0, mipMapCountLimit);

		// sample from specific mip
		float4 pos = float4((ray_pos / WORLD_SIZE) + 0.5, mip_level);

		float4 sampled = tex3Dlod(voxelGiTex, pos);

		// https://kth.diva-portal.org/smash/get/diva2:1886204/FULLTEXT01.pdf
        colour += (1 - opacity) * sampled.w * sampled.xyz;
		opacity += (1 - opacity) * sampled.w;

		total++;

		// exit if the sampled voxel is too opaque (meaning that there would be no point in sampling after this voxel; everything would be occluded)
		if (sampled.w > minOpacityBreak) {
			break;
		}

		// exit if we go outside the bounds of the volume
		if (any(ray_pos < (-WORLD_SIZE * 0.5) | ray_pos > (WORLD_SIZE * 0.5))) {
			break;
		}
    }

    return colour.xyz;
}

[shader("compute")]
[numthreads(32,1,1)]
void CSVoxelizeSecond(uint3 raw_id : SV_DispatchThreadID) {
	if (raw_id.x > indirectArgs[0].w) {
		return;
	}

	uint3 id = index_to_id(readOnlySurfaceVoxels[raw_id.x]);
	uint3 src_tex_id = id.xyz * ambientTextureScalingFactor;

	half3 indirectDiffuse = 0.0;
	float3 position =  (float3)(src_tex_id) * (WORLD_SIZE / TEXTURE_RES) - WORLD_SIZE * 0.5;
	float3 normal = normalsTex[src_tex_id].xyz; 

	int total = 0;

    for (int i = 0; i < randDirs; i++) {
		float3 new_normal = normalize(normal + (hash31(i * 132.123124124) - 0.5) * normalMult);
        indirectDiffuse += TraceStuff(position, new_normal, diffuseHalfAngle, total) * clamp(dot(new_normal, normal), 0, 1);
    }

	indirectDiffuse /= randDirs;

	float4 original = mainTex[src_tex_id];
	ambientTex[id.xyz] = float4(indirectDiffuseFactor * indirectDiffuse + originalDiffuseFactor * original.xyz, original.w);
	// ambientTex[id.xyz] = (float)total / (randDirs * stepCount);
}


RWTexture3D<float4> blurredAmbientTex;

[shader("compute")]
[numthreads(8,8,8)]
void CSAmbientBlur(uint3 id : SV_DispatchThreadID) {
	float4 colour = 0;
	float samples = 0;
	int size = 1;

	for (int x = -size; x <= size; x++) {
		for (int y = -size; y <= size; y++) {
			for (int z = -size; z <= size; z++) {
				float4 loaded = ambientTex[id.xyz + int3(x,y,z) * 1];
				colour += loaded;
				samples += 1.0;
			}
		}	
	}

	blurredAmbientTex[id.xyz] = colour / samples;
}
*/