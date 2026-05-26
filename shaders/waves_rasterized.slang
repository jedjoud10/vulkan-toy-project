#language slang 2026
#include <pbr.slang>
#include <uniform_thing.slang>
#include <ray_stuff_other.slang>
#include <bindless.slang>


struct VertexInput {
    float3 world;
};

struct VertexOutput {
    float3 world_pos;
    float4 sv_position : SV_Position;
};

/*
struct DummyPayload {
    uint2 offset;
}
// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("task")]
[numthreads(1, 1, 1)]
void task_main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    DummyPayload localPayload;
    localPayload.offset = id.xy * 7;
    if (all(gtid == 0)) {
    }
    DispatchMesh(1, 1, 1, localPayload);
}
*/

uint position_to_index(uint x, uint y) {
    return y + x * 9;
}

uint position_to_index_2_bruh(uint x, uint y, uint mult) {
    uint2 p = uint2(x,y) * mult;
    p = clamp(p, 0, 7);

    return p.y + p.x * 9;
}


// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[91], out indices uint3 triangles[128], uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    let uniforms = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);

    float2 meshlet_offset = (float2(id.x, id.y) * 7.0) * 10.0;
    float2 meshlet_offset_2 = (float2(id.x + 1, id.y + 1) * 7.0) * 10.0;
    float2 flat_center = (meshlet_offset+meshlet_offset_2) * 0.5;

    float3 meshlet_center = float3(flat_center.x, 0, flat_center.y); 

    let camera_frustum_planes = uniforms.camera_frustum_planes;
    for (int f = 0; f < 6; f++) {
        float3 plane_normal = camera_frustum_planes[f].xyz;
        float plane_distance = camera_frustum_planes[f].w;

        if ((dot(plane_normal, meshlet_center) + plane_distance) < (-160.0 * 2.0)) {
            return;
        }
    }

    for (uint x = 0; x < 9; x++) {
        for (uint y = 0; y < 9; y++) {
            VertexOutput output;

            float2 p = float2(x, y) * 10 + meshlet_offset;
            float height = snoise(p * 0.008 + uniforms.time * 0.4);
            
            float4 world_position = float4(p.x, height * 0.2, p.y, 1) + float4(hash32(p) - 0.5, 0);

            float4 t = mul(world_position, uniforms.view_projection_matrix);
            t.y = -t.y;
            t.x = -t.x;

            int index = position_to_index(x, y);

            vertices[index].sv_position = t;
            vertices[index].world_pos = world_position.xyz;
        }
    }

    int lod = clamp(floor(distance(uniforms.position.xyz, meshlet_center) / 600), 0, 3);
    int num_tris_axis = (8 >> lod);
    int lod_multiplier = 1 << lod;

    int quad_index = 0;
    for (uint x = 0; x < num_tris_axis; x++) {
        for (uint y = 0; y < num_tris_axis; y++) {
            uint4 corners = uint4(
                position_to_index_2_bruh(x,  y,    lod_multiplier),
                position_to_index_2_bruh(x,  y+1,  lod_multiplier),
                position_to_index_2_bruh(x+1,y,    lod_multiplier),
                position_to_index_2_bruh(x+1,y+1,  lod_multiplier)
            );

            if (hash13(meshlet_center + quad_index * 4.123123) > 0.5) {
                triangles[quad_index * 2] = uint3(corners.x, corners.y, corners.z);
                triangles[quad_index * 2 + 1] = uint3(corners.w, corners.z, corners.y);
            } else {
                triangles[quad_index * 2] = uint3(corners.x, corners.y, corners.w);
                triangles[quad_index * 2 + 1] = uint3(corners.z, corners.x, corners.w);
            }
            quad_index++;
        }
    } 

    SetMeshOutputCounts(91, 64*2);
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target {
    let uniform_buffer = storage_buffers_readonly[UNIFORM_BUFFER_THINGY_IDX].Load<UniformThing>(0);
    let skybox_sampler = sampled_images_cube[SKYBOX_SAMPLER_IMAGE_IDX];
    let clouds_sampler = sampled_images[CLOUDS_SAMPLER_IMAGE_IDX];
    
    float3 world = in_vert.world_pos;
    
    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));
    float3 view = normalize(in_vert.world_pos - uniform_buffer.position.xyz);
    
    float3 reflection = sample_skybox_and_clouds(world, reflect(view, normal), clouds_sampler, skybox_sampler);

    return float4(reflection, 1.0);
}