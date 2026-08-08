#language slang 2026
#include "utils/bindless.slang"
#include "utils/sdf_iq.slang"
#include "utils/sdf_ext.slang"

struct Primitive {
    float3 position;
    uint flags;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id: SV_DispatchThreadID, uniform uint add) {   
    
    /*
    float d = sdFbm(p + float3(uniforms.time * 0.01, 0, 0), p.y - 0.6);

    d = p.y - 0.6;
    d = min(d, sdBase(p));
    */

    /*
    float d = 1000;


    for (int i = 0; i < 10; i++) {
        Primitive prim = storage_buffers_readonly[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS].Load<Primitive>(i * 16);
    
        float primitive_sdf = sdRoundBox(p - prim.position, 0.6, 0.5);
        //float primitive_sdf = length(p - prim.position) - 0.8;
        //float primitive_sdf = sdTorus(p - prim.position, float2(0.5, 0.25));
        
        d = smin(d, primitive_sdf, 0.1);
    }


    //d = p.y - 64;
    storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Store(id.xyz, d * scale);
    */
    

    // [0..TEXTURE_SIZE]
    float3 p = (id.xyz + 0.5);

    // [-TEXTURE_SIZE..TEXTURE_SIZE]
    p = p * 2 - TEXTURE_SIZE;

    // [-MAP_SIZE..MAP_SIZE]
    p *= MAP_SIZE / TEXTURE_SIZE;
    

    let uniforms = load_per_frame_uniforms();
    

    float prev = storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Load(id.xyz).x;

    float3 pos = uniforms.position.xyz + uniforms.forward.xyz * 5.0;

    float sdf = sdRoundBox(p - pos, 0.8, 0.2);
    if (add == 1) {
        prev = min(prev, sdf);
    } else {
        prev = max(prev, -sdf);
    }

    //prev += 0.001;

    storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Store(id.xyz, prev);
}


[shader("compute")]
[numthreads(4, 4, 4)]
void main2(uint3 id: SV_DispatchThreadID) {
    float3 p = (id.xyz + 0.5);
    uniforms = load_per_frame_uniforms();

    // divide by texture size, resulting coord in [0..1] space
    {
        uint w = 0;
        uint h = 0;
        uint d = 0;
        uint lods = 0;
        sampled_images_3d[SDF_TEXTURE2_SAMPLED_IMAGE_IDX].GetDimensions(0, w, h, d, lods);
        p /= float3(w,h,d);
    }

    // remap to [-1..1]
    p = p*2.0 - 1.0;

    // remap to -TREE_SCALE..TREE_SCALE
    p *= TREE_SCALE;

    p *= TREE_SCALE2;

    float2 distance_and_material = tree2(p + float3(0, 4, 0), 0);

    // accomodate distance factor
    float dmult = TREE_SCALE2;

    float2 scaled = distance_and_material / float2(dmult, 1.0);

    storage_images_3d[SDF_TEXTURE2_STORAGE_IMAGE_IDX].Store(id.xyz, float4(scaled, 0));
}

struct aabb {
    float3 min;
    float3 max;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void compute_aabbs(uint3 id: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    float loaded = storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Load(id.xyz).x;

    float dmin = WaveActiveMin(loaded);
    float dmax = WaveActiveMax(loaded);

    float3 k = id.xyz - 64.0;

    float3 aabb_min = 5;
    float3 aabb_max = 0;

    // probably some euclidean distance constants or something
    float theshold = 0.7;

    bool close_enough = loaded < theshold;

    float3 t = gtid;

    // Reduce each component individually
    aabb_min.x = WaveActiveMin(select(close_enough, t.x, 0));
    aabb_min.y = WaveActiveMin(select(close_enough, t.y, 0));
    aabb_min.z = WaveActiveMin(select(close_enough, t.z, 0));

    aabb_max.x = WaveActiveMax(select(close_enough, t.x+1, 0));
    aabb_max.y = WaveActiveMax(select(close_enough, t.y+1, 0));
    aabb_max.z = WaveActiveMax(select(close_enough, t.z+1, 0));
    int3 lid2 = (int3)gid - 16;

    uint aaabb_byte_offset = 3*4*2;

    GroupMemoryBarrierWithWaveSync();

    if (dmax > -theshold && dmin < theshold && all(gtid == 0)) {
        uint count = 0;
    
        storage_buffers[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS+1].InterlockedAdd(0, 1, count);
        
        if (count < 1024*1024*1024) {
            float offset = .0;
            aabb data = aabb(aabb_min + gid * 4 - 64.0 + offset, aabb_max + gid * 4 - 64.0 + offset);
            // aabb data = aabb(gid * 4 - 64.0, gid * 4 + 4 - 64.0);

            storage_buffers[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS].Store(aaabb_byte_offset * count, data.min);
            storage_buffers[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS].Store(aaabb_byte_offset * count + 3*4, data.max);
        }
    }
    
    GroupMemoryBarrierWithWaveSync();
}