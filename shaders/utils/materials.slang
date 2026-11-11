#ifndef MATERIALS
#define MATERIALS
#include "bindless.slang"
#include "triplanar.slang"

struct Material : Layerable {
    uint32_t base_texture_index;

    static Material load(uint32_t material_index) {
        GpuMaterialMetadata meta = storage_buffers_readonly[STORAGE_BUFFER_MATERIALS_BUFFER].Load<GpuMaterialMetadata>(sizeof(GpuMaterialMetadata) * material_index);
        return Material(meta.texture_base_index);
    }

    static Layer sample_triplanar(Material mat, float3 world, float3 normal) {
        float3 diffuse = sample_triplanar(mat.base_texture_index, world, normal);
        float3 arm = sample_triplanar(mat.base_texture_index + 1, world, normal);
        float3 normal = normalize(sample_triplanar_normal(mat.base_texture_index + 2, world, normal));
        return Layer(diffuse, normal, arm);
    }

    Layer apply(float3 world, float3 normal) {
        return Material.sample_triplanar(this, world, normal);
    }
}

struct Layer {
    float3 diffuse;
    float3 normal;
    float3 arm;
}

struct BaseEmptyLayer : Layerable {
    Layer apply(float3 world, float3 normal) {
        return Layer(0, normal, 0);
    }
}

struct Layered<T : Layerable, T2 : Layerable> : Layerable {
    T previous_layers;
    T2 other_layer;
    float weight;

    Layer apply(float3 world, float3 normal) {
        // skip evaluating the bottom layer if we know this layer will be on top
        if (weight == 1.0) {
            return other_layer.apply(world, normal);
        }

        
        // skip evaluating the top layer if we know it will not influence the result
        if (weight == 0.0) {
            return previous_layers.apply(world, normal);
        }

        // over here, still no texture samples done yet, perform both
        Layer b = previous_layers.apply(world, normal);
        Layer a = other_layer.apply(world, normal);
        float3 diffuse = lerp(b.diffuse, a.diffuse, weight);
        float3 arm = lerp(b.arm, a.arm, weight);

        // TODO: this is actually wrong, needs to "re-orient" the top normal based on the bottom normal
        float3 normal = normalize(lerp(b.normal, a.normal, weight));

        return Layer(diffuse, normal, arm);
    }
}

interface Layerable {
    Layer apply(float3 world, float3 normal);
}

Layered<T, N> add_layer<T : Layerable, N : Layerable>(T previous, N wowza, float weight) where T : Layerable {
    return Layered<T, N>(previous, wowza, weight);
}

Layer smush<T : Layerable>(T layered, float3 world, float3 normal) where T : Layerable {
    return layered.apply(world, normal);
}

#endif