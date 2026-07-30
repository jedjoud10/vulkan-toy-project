#ifndef RT
#define RT

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"

struct SparseVoxelOctree {
    uint64_t get_aabb(uint32_t index) {
        return storage_buffers_readonly[STORAGE_BUFFER_SVO_AABB].Load<uint64_t>(index * 8);
    }

    uint64_t get_bitmask(uint32_t index) {
        return storage_buffers_readonly[STORAGE_BUFFER_SVO_BITMASK].Load<uint64_t>(index * 8);
    }

    uint32_t get_index(uint32_t index) {
        return storage_buffers_readonly[STORAGE_BUFFER_SVO_INDEX].Load<uint32_t>(index * 4);
    }
}

SparseVoxelOctree get_bindless_svo() {
    return SparseVoxelOctree();
}

struct Statistics {
    int iterations;
    int num_entered_but_missed;
    
    int buffer_fetches;
    int exit_type;
    int exit_node_height;
    int active_threads_per_call;
    int calls;

    int largest_parent_to_child_index_gap;
}

static const int DEPTH = 7;
static const float EPS = 0.00001;
static const uint32_t BOTTOM_NODE = 0xFFFFFFFF;
static const uint32_t FULL_NODE = 0xFFFFFFFF-1; 
static const int TOTAL_SIZE = 1 << (DEPTH * 2);


// https://tavianator.com/2022/ray_box_boundary.html
bool ray_box_intersection(
    float3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int32_t3 aabb_min,
    int32_t3 aabb_max,
    out float _tmin,
) {
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);


    _tmin = tmin;

    return max(tmin, 0) <= tmax;
}

bool ray_box_intersection3(
    float3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int16_t3 aabb_min,
    int16_t3 aabb_max,
    out float _tmin,
    out int face,
) {
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);


    _tmin = tmin;

    bool3 eqs = max3(k1.x, k1.y, k1.z) == k1;
    face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);

    return max(tmin, 0) <= tmax;
}

bool ray_box_intersection2(
    float3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    float3 aabb_min,
    float3 aabb_max,
) {
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);

    return max(tmin, 0) <= tmax;
}


struct DdaTraversalOutput {
    bool hit;
    //float3 world; // this is not entirely accurate as there can be floating point issues here and there. DO NOT TRUST THIS FOR EXACT RESULTS!!
    //uint3 block_pos; // this is always 100% accurate when we reach the bottom most node of the acceleration structure
    uint specific_block_axis_thing;
    float distance;
    uint8_t face;
}

// TODO: OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE
// 1. optimize occupancy, we still have buffer latency left to hide
// 2. optimize raw speed, it is still relatively slow on my HW
DdaTraversalOutput dda_trace_recurse<let K : int>(
    uint32_t base_index,
    uint32_t actual_child_index,
    float3 ray_dir,
    float3 ray_pos,
    float3 inv_dir,
    int8_t3 dir_sign,
    float distance,
    SparseVoxelOctree svo,
    uint32_t3 parent_world_space_origin,
    uint8_t face,
    inout Statistics stats,
) {
    float tmin = 0;
    int face2 = 0;


    uint64_t raw = 0;

    if (((base_index >> 30) & 1) == 0) {
        raw = svo.get_bitmask(actual_child_index);
        stats.buffer_fetches += 1;
    } else {
        raw = uint64_t.maxValue;
    }

    //base_index &= ~(1 << 30); // get rid of flag bits

    /*
    if (WaveActiveCountBits(true) < 16) {
        return DdaTraversalOutput(false);
    }
    */

    uint32_t voxel_size = (uint32_t)1 << max(K*2, 0);
    
    // do everything in world space first, and then we convert to local space
    // ray will always be in the parent's working space, so we offset by the parent origin and clamp coords to [0..4)
    float3 modified_ray_pos = ((ray_pos + ray_dir * distance - parent_world_space_origin) / (float)voxel_size);
    modified_ray_pos = clamp(modified_ray_pos, EPS, 4 - EPS);
    
    // yes technically you can repr this by a single byte and do packing/unpacking ops
    // I tried that, it did not help perf.
    int8_t3 floored_pos = (int8_t3)floor(modified_ray_pos);
    floored_pos = (int8_t3)clamp(floored_pos, 0, 4);

    float3 side_dist = ((dir_sign * ((floored_pos - modified_ray_pos) + 0.5) + 0.5) * inv_dir);
    bool3 eqs = false;    

    uint min_depth = load_per_frame_uniforms().debug_type;
    
    
    for (uint8_t i = 0; i < 12; i++) {
        stats.iterations += 1;
        
        // we know that floored pos is the local coordinate directly anyways
        uint8_t3 local = (uint8_t3)floored_pos;
        uint8_t bit_index = local.x | local.y << 2 | local.z << 4; 


        stats.calls += 1;
        stats.active_threads_per_call += WaveActiveCountBits((((raw >> bit_index) & 1) == 1));
        


        [branch]
        if ((((raw >> bit_index) & 1) == 1)) {  
            // world space position of the node origin
            uint32_t3 origin = parent_world_space_origin + floored_pos * voxel_size;

            // check if we have reached the bottom most node and exit if so
            [branch]
            if (K > 0 && K > min_depth) {
                // calculate child offset index, which is NOT the same as the bit index
                // we need to count the number of ones that come before (excluding) `bit_index` in `raw` 
                // create a mask for the first `n` bits
                uint64_t mask = ((uint64_t)1 << bit_index) - 1;
                uint32_t child_offset_index = (uint32_t)countbits(mask & raw);


                // POSSIBLE OPTIMIZATION: store a "delta offset" compared to the base node, which could allow us to encode this as a u16 instead...
                //    - we can store the starting index for an entire level (depth=0,1,2,3,...) of the tree in a wayyy smaller buffer and use that as an offsetter as well, since we know that child indices are greater than parent indices!
                // this should allow us to use u16, even though we have an extra buffer fetch, but that one should be cheap because it is depth based (uses `K`, compiler should store that in scalar mem)
                uint32_t actual_child_index = base_index + child_offset_index;
                uint32_t new_base_index = svo.get_index(actual_child_index);
                stats.buffer_fetches += 1;
                stats.largest_parent_to_child_index_gap = max(stats.largest_parent_to_child_index_gap, new_base_index-base_index);

                // check if we reached a full node and if we did, we must somehow calculate block position and world position accurately 
                if (new_base_index == FULL_NODE) {
                    stats.exit_type = 2;
                    stats.exit_node_height = K;

                    float tmin = 0;
                    ray_box_intersection(inv_dir, dir_sign, ray_pos, origin, origin+voxel_size, tmin);
                    //float3 world = ray_pos + ray_dir * (tmin - EPS);
                    //uint3 block_pos = floor(world);

                    // apply "accurate" block pos since we know the bounds of the full node and input face
                    //block_pos[face] = origin[face] + tmp_sign[face] * (voxel_size-1);
                    
                    int3 tmp_sign = max(0, -dir_sign);
                    return DdaTraversalOutput(true, origin[face] + tmp_sign[face] * (voxel_size-1), tmin - EPS, face);
                    //return DdaTraversalOutput(true, world, block_pos, tmin, face);
                }
                
                float distance_to_surface = distance + length(eqs * (side_dist - inv_dir) * voxel_size);
                var result = dda_trace_recurse<K - 1>(new_base_index, actual_child_index, ray_dir, ray_pos, inv_dir, dir_sign, distance_to_surface, svo, origin, face, stats);
                if (result.hit) {
                    return result;
                } else {
                    stats.num_entered_but_missed++;
                }
            } else { 
                float distance_to_surface = distance + length(eqs * (side_dist - inv_dir) * voxel_size);     
                float3 world = ray_pos + ray_dir * (distance_to_surface - EPS);
                stats.exit_type = 3;
                stats.exit_node_height = K;

                int3 tmp_sign = max(0, -dir_sign);
                return DdaTraversalOutput(true, origin[face], distance_to_surface - EPS, face);
            }
        }

        eqs = side_dist.xyz <= min(side_dist.yzx, side_dist.zxy);
        face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);

        // reason these are faster than select is because FMA. thanks d4rkpl4y3r
        side_dist += eqs * inv_dir;
        floored_pos += eqs * dir_sign;

        // exit early if we leave the space
        if (any(floored_pos < 0 | floored_pos > 3)) {
            return DdaTraversalOutput(false);
        }
    }

    return DdaTraversalOutput(false);
}


DdaTraversalOutput trace_shi(
    float3 ray_pos,
    float3 ray_dir,
    SparseVoxelOctree svo,
    inout Statistics iter,
) {
    if (any(ray_pos < 0 | ray_pos >= TOTAL_SIZE)) {
        return DdaTraversalOutput(false);
    }

    float3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    return dda_trace_recurse<DEPTH - 1>(1, 0, ray_dir, ray_pos, inv_dir, dir_sign, 0.0, svo, 0, 0, iter);
}

/*
struct Data {
    int3 min_bounds;
    int3 max_bounds;
    bool leaf;
}

Data start_from_root_go_down(
    int3 position,
    inout int wowza,
) {
    SparseVoxelOctree svo = get_bindless_svo();

    // start from root
    uint32_t  base_index = 1;
    uint32_t  actual_child_index = 0;

    uint64_t raw = 0;
    int3 parent_world_space_origin = 0;
    uint16_t voxel_size = 0;


    for (int l = DEPTH-1; l >= 0; l--) {
        wowza++;
        voxel_size = (uint16_t)1 << max(l*2, 0);
        if (((base_index >> 30) & 1) == 0) {
            raw = svo.get_bitmask(actual_child_index);
        } else {
            raw = uint64_t.maxValue;
        }

        // calculate the world position relative to the node we are in
        // required to calculate bit index
        uint8_t3 local = (uint8_t3)((position - parent_world_space_origin) / voxel_size);
        uint8_t bit_index = local.x | local.y << 2 | local.z << 4; 
        uint16_t3 origin = parent_world_space_origin + local * voxel_size;
        if (raw == 0) {
            return Data(parent_world_space_origin, parent_world_space_origin + voxel_size, false);
        }

        [branch]
        if ((((raw >> bit_index) & 1) == 1)) {  
            parent_world_space_origin = parent_world_space_origin + local * voxel_size;

            // check if we have reached the bottom most node and exit if so
            [branch]
            if (l > 0) {
                // calculate child offset index, which is NOT the same as the bit index
                // we need to count the number of ones that come before (excluding) `bit_index` in `raw` 
                // create a mask for the first `n` bits
                uint64_t mask = ((uint64_t)1 << bit_index) - 1;
                uint32_t child_offset_index = (uint32_t)countbits(mask & raw);


                // POSSIBLE OPTIMIZATION: store a "delta offset" compared to the base node, which could allow us to encode this as a u16 instead...
                //    - we can store the starting index for an entire level (depth=0,1,2,3,...) of the tree in a wayyy smaller buffer and use that as an offsetter as well, since we know that child indices are greater than parent indices!
                // this should allow us to use u16, even though we have an extra buffer fetch, but that one should be cheap because it is depth based (uses `K`, compiler should store that in scalar mem)
                uint32_t new_actual_child_index = base_index + child_offset_index;
                uint32_t new_base_index = svo.get_index(new_actual_child_index);

                // check if we reached a full node and if we did, we must somehow calculate block position and world position accurately 
                if (new_base_index == FULL_NODE) {
                    return Data(origin, origin + voxel_size, true);
                }
                
                base_index = new_base_index; 
                actual_child_index = new_actual_child_index;
            } else { 
                return Data(position, position + 1, true);
            }
        } else {
            break;
        }
    }

    return Data(parent_world_space_origin, parent_world_space_origin + voxel_size, false);
}
*/

#endif