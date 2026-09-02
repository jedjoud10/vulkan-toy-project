#ifndef SDF_EXTENDED
#define SDF_EXTENDED

#include "sdf_iq.slang"

[Differentiable]
float dsmoothstep(float a, float b, float x) {
    let k = saturate((x - a) / (b - a));
    return 3*k*k - 2*k*k*k;
}

[Differentiable]
float tree1(float3 lp) {
    float3 rotated = lp.xzy;
    
    // base trunk
    float d = sdHexPrism(rotated - float3(0, 0, -3.0), float2(0.5, 5));
    
    d = min(d, sdRoundBox(rotated - float3(0, 0, 0.5) + float3(sin(uniforms.time - 3.213), cos(uniforms.time + 4.534), 0) * 0.1, float3(2.0, 2.0, 0.5), 0.2));
    
    d = min(d, sdRoundBox(rotated - float3(0, 0, 4.2) + float3(sin(uniforms.time), cos(uniforms.time), 0) * 0.1, float3(1.5, 1.5, 3.5), 0.2));
    /*
    d = min(d, sdHexPrism(rotated - float3(0, 0, 0.5) + float3(sin(uniforms.time - 3.213), cos(uniforms.time + 4.534), 0) * 0.1, float2(2.0, 0.5)));
    d = min(d, sdHexPrism(rotated - float3(0, 0, 4.5) + float3(sin(uniforms.time), cos(uniforms.time), 0) * 0.1, float2(1.5, 3.5)));
    */
    // d = min(d, sdHexPrism(rotated - float3(0, 0, 1.1), float2(0.6, 2.0)));

    // d += noise(rotated * 2.2) * 0.1;

    return d;
}

float2 tree2(float3 lp, float instance_index) {
    // half extent
    float box_height = 4;
    
    // branch space
    float3 bp = lp - float3(0, -2.0 * box_height, 0);
    float3 bp2 = lp - float3(0, -2.0 * box_height, 0);
    
    float d = 10000;
    float thickness = 0.8;
    
    float3x3 local_rot = AngleAxis3x3(0.2 + sin(instance_index) * 0.1, float3(1, 0, -1));

    for (int i = 0; i < 4; i++) {
        // non smooth abs for branches
        bp.xz = abs(bp.xz);

        // smooth abs for leaves
        bp2.xz = sqrt(pow(bp2.xz, 2.0) + 0.35);

        // box is center at (0, 1, 0) in local branch space
        // the origin (bp = (0, 0, 0)) is the BOTTOM of the box
        d = min(d, sdBox(bp - float3(0, box_height, 0), float3(thickness, box_height, thickness))) - noise(bp * 2.0 * float3(1, 3, 1)) * 0.02;

        // move UP to reach end of box
        bp -= float3(0, 2 * box_height, 0);
        bp2 -= float3(0, 2 * box_height, 0);

        // rotate space
        bp = mul(bp, local_rot);
        bp2 = mul(bp2, local_rot);
        
        // make next branches smaller and shorter
        thickness *= 0.8  + cos(34.2 * instance_index) * 0.1;
        box_height *= 0.6 + cos(3.2  * instance_index) * 0.1;
    }

    float id = 0;

    float sphere = length(bp2 - float3(0, 2, 0)) - 2.0;
    if (sphere < d) {
        d = sphere;
        id = 1;
    }
    
    //d = smin(d, , 0.2);

    return float2(d, id);
}

const static float MAP_SIZE = 32;
const static float TEXTURE_SIZE = 256;
const static float SMOOTHING_FACTOR_MAP_TEXTURE = 0.05;


// p is in world space [-MAP_SIZE..MAP_SIZE]
float tex_sample(float3 p) {
    // [0..MAP_SIZE]
    p = p * 0.5 + MAP_SIZE * 0.5;

    // [0..TEXTURE_SIZE]
    p *= TEXTURE_SIZE / MAP_SIZE;

    // [0..1]
    p /= TEXTURE_SIZE;

    return sampled_images_3d[SDF_TEXTURE_SAMPLED_IMAGE_IDX].Sample(samplers[LINEAR_UNCAMPLED_SAMPLER_IDX], p).x;
}

[BackwardDerivativeOf(tex_sample)]
void bwd_tex_sample(inout DifferentialPair<float3> p_pair, float dOut)
{
    float eps = 0.1;
    float x = tex_sample(p_pair.p - float3(eps, 0, 0));
    float y = tex_sample(p_pair.p - float3(0, eps, 0));
    float z = tex_sample(p_pair.p - float3(0, 0, eps));
    float b = tex_sample(p_pair.p);


    p_pair = diffPair<float3>(p_pair.p, (float3(b-x, b-y, b-z) / eps) * dOut);
}

const static float TREE_SCALE = 6;
const static float TREE_SCALE2 = 2.5;



// p is in local space, [-TREE_SCALE..TREE_SCALE]
float2 sample_tree(float3 p) {
    uint w = 0;
    uint h = 0;
    uint d = 0;
    uint lods = 0;
    sampled_images_3d[SDF_TEXTURE2_SAMPLED_IMAGE_IDX].GetDimensions(0, w, h, d, lods);

    // map to [-1..1]
    p /= TREE_SCALE;

    // map to 0..1
    p = p * 0.5 + 0.5;

    return sampled_images_3d[SDF_TEXTURE2_SAMPLED_IMAGE_IDX].Sample(samplers[LINEAR_CLAMPED_SAMPLER_IDX], p).xy;
}

[BackwardDerivativeOf(sample_tree)]
void bwd_sample_tree(inout DifferentialPair<float3> p_pair, float2 dOut)
{
    float eps = 0.01;
    float x = sample_tree(p_pair.p - float3(eps, 0, 0)).x;
    float y = sample_tree(p_pair.p - float3(0, eps, 0)).x;
    float z = sample_tree(p_pair.p - float3(0, 0, eps)).x;
    float b = sample_tree(p_pair.p).x;


    p_pair = diffPair<float3>(p_pair.p, (float3(b-x, b-y, b-z) / eps) * dOut.x);
}

#endif