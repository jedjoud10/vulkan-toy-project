#ifndef SDF_CUSTOM
#define SDF_CUSTOM

#include "other.slang"
#include "noises.slang"
#include "bindless.slang"
#include "sdf_iq.slang"
#include "sdf_ext.slang"

enum SdfType {
    Box,
    BoxWithCutout,
    Rock,
    Interesting1,
    Tree,
}

float get_sdf_from_type(float3 local, float3 global, SdfType type, uint object_index) {

    switch (type) {
        case SdfType.Box:
            return sdBox(local, 0.5);
        case SdfType.BoxWithCutout:
            float box = sdBox(local, 0.99);
            box = max(box, -sdCylinder(local.xyz, float3(0.0, 0.0, 0.6)));
            box = max(box, -sdCylinder(local.xzy, float3(0.0, 0.0, 0.6)));
            box = max(box, -sdCylinder(local.yxz, float3(0.0, 0.0, 0.6)));
            
            box = max(box, sdSphere(local, 1.0));            
            return box;
        case SdfType.Rock:
            float s = sdBox(local, 0.8);
            float blending = noise(global * 10) * 0.1 + 0.01;

            [ForceUnroll]
            for (int i = 0; i < 6; i++) {
                int seed = i + object_index * 52;
                float3 normal = normalize(float3(sin(seed * 40.21), sin(seed * 10.21), sin(seed * -20.43)));
                if (length(normal) > 0.01) {
                    float k = dot(normalize(normal), local) - 0.5;
                    s = smax(s, k, blending);
                }
            }

            return s;
        case SdfType.Interesting1:
            s = max(sdBox(local, 0.8), sdCylinder(local, float3(0, 0, 0.4)));

            float3 k = mul(local, AngleAxis3x3(uniforms.time, normalize(hash31(object_index))));
            k.x = abs(k.x);
            k = k - float3(0.6, 0.4, 0.0);
            s = min(sdSphere(k, 0.2), s);

            s = min(max(sdSphere(local - float3(0, -0.3, 0), 0.6), sdSphere(local - float3(0, 0.3, 0), 0.6)), s);

            s += sin(local.y*30 + uniforms.time*4.0)*0.01;

            return s;

        case SdfType.Tree:
            return tree2(local*8, object_index).x/8;
    }

    return 0.0;
}

#endif