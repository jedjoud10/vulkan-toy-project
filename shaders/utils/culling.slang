#include "bindless.slang"

float box_aabb_half_extent_to_sphere_radius(float half_extent) {
    return half_extent * 1.41221; // sqrt(2x^2) / x ratio
}

bool frustum_cull_sphere_is_visible(float3 center, float radius) {
    let uniforms = load_per_frame_uniforms();
    
    // toggle debug frustum culling
    if ((uniforms.toggles_bitmask & 1) == 1) {
        radius = 0.40;
    }

    let camera_frustum_planes = uniforms.camera_frustum_planes;
    for (int f = 0; f < 6; f++) {
        float3 plane_normal = camera_frustum_planes[f].xyz;
        float plane_distance = camera_frustum_planes[f].w;

        if ((dot(plane_normal, center) + plane_distance) < -radius) {
            return false;
        }
    }

    return true;
}