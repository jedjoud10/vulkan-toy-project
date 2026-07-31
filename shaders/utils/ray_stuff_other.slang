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

struct Configuration {
}

static const int DEPTH = 7;
static const float EPS = 0.00001;
static const uint32_t BOTTOM_NODE = 0xFFFFFFFF;
static const uint32_t FULL_NODE = 0xFFFFFFFF-1; 
static const int TOTAL_SIZE = 1 << (DEPTH * 2);

bool ray_box_intersection(
    float3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int32_t3 aabb_min,
    int32_t3 aabb_max,
) {
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);

    return max(tmin, 0) <= tmax;
}


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

// u64 bitmask ordering: 
//    for x in  0..4i32 {
//        for y in 0..4i32 {
//            for z in 0..4i32 {
//                let i = (z + y * 4 + x * 4 * 4) as u32;



// creates a 4-bit wide bitmask that state if any of the 4 x layers have any bits
uint8_t extract_x_layer_bits(uint64_t bitmask) {
    const static uint64_t MASK_16_BITS_SET = 0b1111_1111_1111_1111;
    
    uint8_t layer1 = ((bitmask & MASK_16_BITS_SET) != 0); 
    uint8_t layer2 = (((bitmask >> 16) & MASK_16_BITS_SET) != 0); 
    uint8_t layer3 = (((bitmask >> 32) & MASK_16_BITS_SET) != 0); 
    uint8_t layer4 = (((bitmask >> 48) & MASK_16_BITS_SET) != 0); 

    return layer1 | layer2 << 1 | layer3 << 2 | layer4 << 3;
}

// creates a 4-bit wide bitmask that state if any of the 4 y layers have any bits
uint8_t extract_y_layer_bits(uint64_t bitmask) {
    const static uint64_t LAYER_Y_INTERLEAVING_MASK = 0b1111 | (0b1111 << 16) | (0b1111 << 32) | (0b1111 << 48);
    
    uint8_t layer1 = ((bitmask & LAYER_Y_INTERLEAVING_MASK) != 0); 
    uint8_t layer2 = (((bitmask >> 4) & LAYER_Y_INTERLEAVING_MASK) != 0); 
    uint8_t layer3 = (((bitmask >> 8) & LAYER_Y_INTERLEAVING_MASK) != 0); 
    uint8_t layer4 = (((bitmask >> 12) & LAYER_Y_INTERLEAVING_MASK) != 0); 

    return layer1 | layer2 << 1 | layer3 << 2 | layer4 << 3;
}

// creates a 4-bit wide bitmask that state if any of the 4 z layers have any bits
uint8_t extract_z_layer_bits(uint64_t bitmask) {
    const static uint64_t LAYER_Z_INTERLEAVING_MASK = 0b0001 | (0b0001 << 4) | (0b0001 << 8) | (0b0001 << 12);
    
    uint8_t layer1 = ((bitmask & LAYER_Z_INTERLEAVING_MASK) != 0); 
    uint8_t layer2 = (((bitmask >> 1) & LAYER_Z_INTERLEAVING_MASK) != 0); 
    uint8_t layer3 = (((bitmask >> 2) & LAYER_Z_INTERLEAVING_MASK) != 0); 
    uint8_t layer4 = (((bitmask >> 3) & LAYER_Z_INTERLEAVING_MASK) != 0); 

    return layer1 | layer2 << 1 | layer3 << 2 | layer4 << 3;
}


// holy fucking shit this is cursed
const static uint8_t[] lookup = {
    // 0b0000: INVALID!!!
    0b10000,

    // 0b0001:
    0 | (0 << 2),

    // 0b0010:
    1 | (1 << 2),

    // 0b0011:
    0 | (1 << 2),
    
    // 0b0100:
    2 | (2 << 2),

    // 0b0101:
    0 | (2 << 2),

    // 0b0111:
    0 | (2 << 2),

    // 0b1001:
    0 | (3 << 2),

    // 0b1010:
    1 | (3 << 2),

    // 0b1011:
    0 | (3 << 2),
    
    // 0b1100:
    2 | (3 << 2),

    // 0b1101:
    0 | (3 << 2),

    // 0b1111:
    0 | (3 << 2),
};

// layer_bits is 4 bit wide bitmask
// returns 1D bounds. min (from least sig. bit) is stored in the first 2 bits. max (from least sig. bit) is stored in the next 2 bits
// min is inclusive. max is inclusive as well. 5th bit states invalidity
uint8_t get_max_min_layer_bits(uint8_t layer_bits) {
    // TODO: DONT DO THIS. WHY IS THERE NO LZCNT OP
    return lookup[layer_bits];
}

struct RelativePackedAABB {
    // we just need 12 bits in total
    // 2 bits per axis, 3 axis, min+max = 12 bits
    // first 6 bits are the x,y,z coords of min (in that order)
    // next 6 bits are the x,y,z coords of max (in that order)
    uint16_t data;  

    property uint8_t3 min {
        get { return uint8_t3(data & 0b11, (data >> 2) & 0b11, (data >> 4) & 0b11); }
	}

    property uint8_t3 max {
        get { return uint8_t3((data >> 6) & 0b11, (data >> 8) & 0b11, (data >> 10) & 0b11); }
	}
}

RelativePackedAABB extract_aabb_bitmask(uint64_t bitmask) {
    uint8_t x = extract_x_layer_bits(bitmask);
    uint8_t y = extract_y_layer_bits(bitmask);
    uint8_t z = extract_z_layer_bits(bitmask);

    // first 2 bits are min, next 2 bits are max (both inclusive)
    uint8_t bounds_x = get_max_min_layer_bits(x);
    uint8_t bounds_y = get_max_min_layer_bits(y);
    uint8_t bounds_z = get_max_min_layer_bits(z);

    // max bounds testing
    // bounds_x = 0b11_00;
    // bounds_y = 0b11_00;
    // bounds_z = 0b11_00;    

    uint16_t min_part = bounds_x & 0b11 | (bounds_y & 0b11) << 2 | (bounds_z & 0b11) << 4;
    uint16_t max_part = ((bounds_x & 0b1100) >> 2) | ((bounds_y & 0b1100) >> 2) << 2 | ((bounds_z & 0b1100) >> 2) << 4;

    return RelativePackedAABB(min_part | max_part << 6);
}


struct DdaTraversalOutput {
    bool hit;
    //float3 world; // this is not entirely accurate as there can be floating point issues here and there. DO NOT TRUST THIS FOR EXACT RESULTS!!
    //uint3 block_pos; // this is always 100% accurate when we reach the bottom most node of the acceleration structure
    uint specific_block_axis_thing;
    float distance;
    uint8_t face;
    uint depth;
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
    Configuration config,
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

    UniformThing uniforms = load_per_frame_uniforms();
    uint min_depth = uniforms.debug_type;
    bool aabb_test_toggle = (uniforms.toggles_bitmask & 2) != 0;

    if (K > 2 && aabb_test_toggle) {
        RelativePackedAABB babaisyou = extract_aabb_bitmask(raw);
        int32_t3 aabb_global_min = babaisyou.min * (voxel_size << 0) + parent_world_space_origin;
        int32_t3 aabb_global_max = babaisyou.max * (voxel_size << 0) + parent_world_space_origin;
        // int32_t3 aabb_global_min = int32_t3(0) * (voxel_size << 2) + parent_world_space_origin;
        // int32_t3 aabb_global_max = int32_t3(3) * (voxel_size << 2) + parent_world_space_origin;
        // aabb_global_min -= voxel_size;
        // aabb_global_max += voxel_size;
        
        float tmin = 0;
        if (ray_box_intersection(inv_dir, dir_sign, ray_pos, aabb_global_min, aabb_global_max, tmin)) {
            //distance = max(tmin, distance);
        } else {
            return DdaTraversalOutput(false);
        }
    }
    
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
            bool loaded_in = length((origin + voxel_size * 0.5) - ray_pos) < voxel_size * 200.5 + hash13(ray_dir * 40.12314 * float3(10.4324, -50.23423, 60.13)) * 10 * voxel_size;

            [branch]
            if (K > 0) {
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
                    return DdaTraversalOutput(true, origin[face] + tmp_sign[face] * (voxel_size-1), tmin - EPS, face, K);
                    //return DdaTraversalOutput(true, world, block_pos, tmin, face);
                }
                
                if (loaded_in || K > min_depth) {
                    float distance_to_surface = distance + length(eqs * (side_dist - inv_dir) * voxel_size);
                    var result = dda_trace_recurse<K - 1>(new_base_index, actual_child_index, ray_dir, ray_pos, inv_dir, dir_sign, distance_to_surface, svo, config, origin, face, stats);
                    if (result.hit) {
                        return result;
                    } else {
                        stats.num_entered_but_missed++;
                    }
                }                
            } else { 
                float distance_to_surface = distance + length(eqs * (side_dist - inv_dir) * voxel_size);     
                float3 world = ray_pos + ray_dir * (distance_to_surface - EPS);
                stats.exit_type = 3;
                stats.exit_node_height = K;

                int3 tmp_sign = max(0, -dir_sign);
                return DdaTraversalOutput(true, origin[face], distance_to_surface - EPS, face, 0);
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
    Configuration config,
    SparseVoxelOctree svo,
    inout Statistics iter,
) {
    if (any(ray_pos < 0 | ray_pos >= TOTAL_SIZE)) {
        return DdaTraversalOutput(false);
    }

    float3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    return dda_trace_recurse<DEPTH - 1>(1, 0, ray_dir, ray_pos, inv_dir, dir_sign, 0.0, svo, config, 0, 0, iter);
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