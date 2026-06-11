#language slang 2026
#include "utils/pbr.slang"
#include "utils/triplanar.slang"
#include "utils/bindless.slang"

struct PushConstants {
    float4x4 object_to_world;
    uint albedo_sampled_image_index;
}

struct VertexInput {
    float3 vertex;
    float3 normal;
    float2 uv;
};

struct VertexOutput {
    float3 world_pos;
    float3 world_normal;
    float2 uv;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vert_main(VertexInput input, uniform PushConstants pc) {
    VertexOutput output;

    let uniforms = load_per_frame_uniforms();

    float4 world_position = mul(float4(input.vertex, 1), pc.object_to_world);
    float4 world_normal = mul(float4(input.normal, 0), pc.object_to_world);
    


    float4 t = mul(world_position, uniforms.view_projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    output.sv_position = t;
    output.world_pos = world_position.xyz;
    output.world_normal = world_normal.xyz;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert, uniform PushConstants pc) : SV_Target {
    let uniforms = load_per_frame_uniforms();

    // I eventually need to understand why this works but I don't...
    // compute tangent and bitangent from derivatives of position and UV
    float3 dp1 = ddx(in_vert.world_pos);
    float3 dp2 = ddy(in_vert.world_pos);
    float2 duv1 = ddx(in_vert.uv);
    float2 duv2 = ddy(in_vert.uv);

    // solve for tangent and bitangent
    float3 normal = normalize(cross(dp1, dp2));
    float3 tangent = normalize(duv2.y * dp1 - duv1.y * dp2);
    float3 bitangent = normalize(-duv2.x * dp1 + duv1.x * dp2);
    float3x3 tbn = float3x3(tangent, bitangent, normal);
            
    float3 world = in_vert.world_pos;
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);  

    if ((uniforms.toggles_bitmask & 1) == 1) {
        normal = normalize(in_vert.world_normal);
    }

    // sample textures
    let sampler = samplers[NEAREST_SAMPLER_IDX];
    float3 diffuse = sampled_images[pc.albedo_sampled_image_index].Sample(sampler, in_vert.uv).xyz;
    float3 arm = sampled_images[pc.albedo_sampled_image_index+1].Sample(sampler, in_vert.uv).xyz;
    
    // calculate world space normal perturbed by sampled tangent normal map
    if ((uniforms.toggles_bitmask & 2) == 2) {
        float3 tangent_normal = sampled_images[pc.albedo_sampled_image_index+2].Sample(sampler, in_vert.uv).xyz * 2.0 - 1.0;
        normal = normalize(mul(tangent_normal, tbn)); 
    }

    // construct PBR data
    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, arm);
    CameraData camera_data = CameraData(view);
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    return float4(lighting2(pbr_surface_data, camera_data, sun_data), 1);
}

/*
[shader("fragment")]
float4 frag_main(VertexOutput in_vert, uniform PushConstants pc) : SV_Target {
    let uniforms = load_per_frame_uniforms();
    let sampler = samplers[NEAREST_SAMPLER_IDX];

    // I eventually need to understand why this works but I don't...
    // compute tangent and bitangent from derivatives of position and UV
    float3 dp1 = ddx(in_vert.world_pos);
    float3 dp2 = ddy(in_vert.world_pos);
    float2 duv1 = ddx(in_vert.uv);
    float2 duv2 = ddy(in_vert.uv);

    // Solve for tangent and bitangent
    float3 normal = normalize(cross(dp1, dp2));
    float3 tangent = normalize(duv2.y * dp1 - duv1.y * dp2);
    float3 bitangent = normalize(-duv2.x * dp1 + duv1.x * dp2);
    float3x3 tbn = float3x3(tangent, bitangent, normal);
        
    if (uniforms.debug_type == 1) {
        return float4(normalize(in_vert.world_normal), 1.0);
    } else if (uniforms.debug_type == 2) {
        return float4(normalize(tangent), 1.0);
    } else if (uniforms.debug_type == 3) {
        return float4(in_vert.world_pos, 1.0);
    }
    
    
    float3 world = in_vert.world_pos;

    normal = 0;
    if ((uniforms.toggles_bitmask & 2) == 2) {
        float3 tangent_normal = sampled_images[pc.albedo_sampled_image_index+3].Sample(sampler, in_vert.uv).xyz;
        tangent_normal = tangent_normal * 2.0 - 1.0;   
        normal = normalize(mul(tangent_normal, tbn)); 
    } else {
        normal = normalize(in_vert.world_normal);
    }

    RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> sq;
    let flags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE;


    
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);  

    float3 ray_pos = in_vert.world_pos;
    float3 ray_dir = reflect(-view, normalize(normal));

    float factor = 1.0;
    for (int i = 0; i < 8; i++) {
        factor = 1.0 - (float)i / 8.0;
        if ((uniforms.toggles_bitmask & 1) == 1) {
            RayDesc desc;

            desc.Origin = ray_pos;
            desc.Direction = ray_dir;
            desc.TMax = 100;
            desc.TMin = 0.001;
            
            sq.TraceRayInline(acceleration_structures[0], flags, 0xFF, desc);
            
            while (sq.Proceed()) {
                if(sq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
                    sq.CommitNonOpaqueTriangleHit();  // forcing to be opaque
            }
        
            
            if (sq.CommittedStatus() == COMMITTED_NOTHING) {
                return float4(sample_skybox_and_clouds(ray_pos, ray_dir) * factor, 1.0);
            }
            
            
            // https://www.khronos.org/blog/introducing-vulkan-ray-tracing-position-fetch-extension
            float3[3] tris = sq.CommittedGetIntersectionTriangleVertexPositions();
            
            world = ray_pos + ray_dir * sq.CommittedRayT();
            normal = -normalize(cross(tris[1] - tris[0], tris[2] - tris[0]));
            normal = mul(sq.CommittedObjectToWorld3x4(), float4(normal, 0.0)).xyz;
            view = -ray_dir;  

            ray_pos = world;
            ray_dir = reflect(-view, normalize(normal));
        }
    }

    // float3 diffuse = sample_triplanar_albedo(pc.albedo_sampled_image_index, world, normal);
    float3 diffuse = sampled_images[pc.albedo_sampled_image_index].Sample(sampler, in_vert.uv).xyz;
    float roughness = sampled_images[pc.albedo_sampled_image_index+1].Sample(sampler, in_vert.uv).x;
    float metallic = sampled_images[pc.albedo_sampled_image_index+2].Sample(sampler, in_vert.uv).x;

    float ao = 1.0;
    float shadow = (1.0-clouds(uniforms.sun.xyz, world).a) * hw_trace_shadow(world + normal * 0.002); 

    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
    CameraData camera_data = CameraData(view);

    // approximate sun color based on height
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    return float4(lighting2(pbr_surface_data, camera_data, sun_data) * factor, 1);
}
*/

/*
[shader("fragment")]
float4 frag_main(VertexOutput in_vert, uniform PushConstants pc) : SV_Target {
    let uniforms = load_per_frame_uniforms();
    
    float3 world = in_vert.world_pos;
    float3 normal = -normalize(cross(ddy(in_vert.world_pos), ddx(in_vert.world_pos)));
    float3 view = -normalize(in_vert.world_pos - uniforms.position.xyz);

    float3 diffuse = sample_triplanar_albedo(pc.albedo_sampled_image_index, world, normal);
    float roughness = 0.8;
    float metallic = 0.2;
    float ao = hw_ao(world, normal);
    float shadow = (1.0-clouds(uniforms.sun.xyz, world).a) * hw_trace_shadow(world); 

    float3 ray_pos = uniforms.position.xyz;

    PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
    CameraData camera_data = CameraData(view);

    // approximate sun color based on height
    float3 sun = uniforms.sun.xyz;
    float3 sun_colour = sun_colour(sun.xyz);
    SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);

    if (uniforms.debug_type == 0) {
        return float4(lighting(pbr_surface_data, camera_data, sun_data), 1);
    }

    return float4(1);
}
*/