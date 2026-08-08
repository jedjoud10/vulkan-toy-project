#language slang 2026
#include "utils/bindless.slang"
#include "utils/sdf_iq.slang"

struct Primitive {
    float3 position;
    uint flags;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id: SV_DispatchThreadID) {
    float scale = 0.25;
    
    float3 p = (id.xyz - 64.0 + 0.5) * scale;
    let uniforms = load_per_frame_uniforms();
    
    /*
    float d = sdFbm(p + float3(uniforms.time * 0.01, 0, 0), p.y - 0.6);

    d = p.y - 0.6;
    d = min(d, sdBase(p));
    */

    float d = 1000;


    for (int i = 0; i < 10; i++) {
        Primitive prim = storage_buffers_readonly[STORAGE_BUFFER_TERRAIN_AABB_BOUNDS].Load<Primitive>(i * 16);
    
        float primitive_sdf = sdRoundBox(p - prim.position, 1.0, 0.5);
        //float primitive_sdf = length(p - prim.position) - 0.8;
        //float primitive_sdf = sdTorus(p - prim.position, float2(0.5, 0.25));
        
        d = smin(d, primitive_sdf, 0.1);
    }


    //d = p.y - 64;
    storage_images_3d_float[SDF_TEXTURE_STORAGE_IMAGE_IDX].Store(id.xyz, d * scale);
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