#language slang 2026
#include "utils/pbr.slang"
#include "utils/bindless.slang"
#include "utils/triplanar.slang"

struct PushConstants {
    float3 *vertex_buffer_address;
    uint *index_buffer_address;

    uint draw_index;
    uint triangle_count_mesh;
};

struct VertexOutput {
    float3 world_pos;
    //float3 normal;
    float3 colour;
    float4 sv_position : SV_Position;
};

groupshared Atomic<uint> loaded_triangles_in_group;

static const float4 positions[3] = {
    float4(0.0, -1.0, 0.0, 1.0),
    float4(-1.0, 1.0, 0.0, 1.0),
    float4(1.0, 1.0, 0.0, 1.0)
};

// https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void mesh_main(out vertices VertexOutput vertices[96], out indices uint3 triangles[32], uint3 id: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID, uniform PushConstants pc) {
    /*
    vertices[0] = VertexOutput(0, 0, positions[0]);
    vertices[1] = VertexOutput(0, 0, positions[1]);
    vertices[2] = VertexOutput(0, 0, positions[2]);

    triangles[0] = uint3(2, 1, 0);
    SetMeshOutputCounts(3, 1);
    return;
    */

    if (gtid.x == 0) {
        loaded_triangles_in_group = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    if (id.x < pc.triangle_count_mesh) {
        loaded_triangles_in_group.add(1, MemoryOrder.Relaxed);
    }

    GroupMemoryBarrierWithGroupSync();

    if (id.x < pc.triangle_count_mesh) {
        // load triangle from index buffer
        // this triangle indexes into the global vertex buffer
        uint3 triangle = ((uint3*)pc.index_buffer_address)[id.x];

        // "local" in terms of the triangle (so in range in [0..2])
        for (int local_vertex_index = 0; local_vertex_index < 3; local_vertex_index++) {
            // using thread ID, load global vertex
            // yes this will duplicate a lot of the vertex transformation stuff
            float3 loaded_vertex = (((float3 *)pc.vertex_buffer_address)[triangle[local_vertex_index]]);

            VertexOutput output;

            float3 world = loaded_vertex;

            let uniforms = load_per_frame_uniforms();

            float4 world_position = float4(world, 1.0);

            float4 t = mul(world_position, uniforms.view_projection_matrix);
            t.y = -t.y;
            t.x = -t.x;

            output.sv_position = t;
            output.world_pos = world_position.xyz;
            output.colour = hash33(gid);
            vertices[local_vertex_index + gtid.x * 3] = output;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (id.x < pc.triangle_count_mesh) {
        triangles[gtid.x] = uint3(0, 1, 2) + gtid.x * 3;
    }    

    GroupMemoryBarrierWithGroupSync();

    let generated_triangles = loaded_triangles_in_group.load(MemoryOrder.Relaxed);

    GroupMemoryBarrierWithGroupSync();

    SetMeshOutputCounts(generated_triangles*3, generated_triangles);
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert, uniform PushConstants pc) : SV_Target {
    let uniforms = load_per_frame_uniforms();

    if ((uniforms.toggles_bitmask & 8) == 8) {
        // return float4(hash31((float)pc.draw_index * 123.456), 1);
        return float4(in_vert.colour, 1);
    }

    uint material_base_texture_index = storage_buffers_readonly[STORAGE_BUFFER_MATERIALS_BUFFER].Load<uint>(4 * 0);

    float3 dp1 = ddx(in_vert.world_pos);
    float3 dp2 = ddy(in_vert.world_pos);
    float3 normal = normalize(cross(dp1, dp2));            
    float3 world = in_vert.world_pos;
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);  

    
    float3 diffuse = sample_triplanar(material_base_texture_index, world, normal);
    float3 arm = sample_triplanar(material_base_texture_index+1, world, normal);
    normal = normalize(sample_triplanar_normal(material_base_texture_index+2, world, normal));

    

    
    // construct PBR data
    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, AoRoughnessMetallic(arm));
    CameraData camera_data = CameraData(view);
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    // return float4(lighting3<1>(pbr_surface_data, camera_data, sun_data), 1);
    float3 rendered = lighting2(pbr_surface_data, camera_data, sun_data);
    // float3 rendered = hw_trace_shadow_light(world, sun);
    return float4(rendered, 1);
}