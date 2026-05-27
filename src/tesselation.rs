pub fn precompute_tesselation_buffer() {
    let depths = 4;
    let max_verts = 66;
    let max_tris = 64;
    let mut tess_vert_count = format!("const static uint[{depths}] TESS_VERT_COUNT = {{");
    let mut tess_tri_count = format!("const static uint[{depths}] TESS_TRI_COUNT = {{");
    let mut tess_verts_weights = format!("const static float3[{max_verts}][{depths}] TESS_VERTICES_WEIGHTS = {{");
    let mut tess_tris = format!("const static uint3[{max_tris}][{depths}] TESS_TRIANGLES = {{");

    for tesselation_depth in 0..depths {
        let mut verts = vec![
            vek::Vec3::new(1f32, 0f32, 0f32),
            vek::Vec3::new(0f32, 1f32, 0f32),
            vek::Vec3::new(0f32, 0f32, 1f32)
        ];

        let mut previous_triangles = vec![vek::Vec3::new(0,1,2usize)];
        let mut next_triangles = vec![vek::Vec3::new(0,1,2usize)];
        
        for _ in 0..tesselation_depth {
            next_triangles.clear();

            for base_triangle_indices in previous_triangles.iter() {

                let v1 = (verts[base_triangle_indices.x] + verts[base_triangle_indices.y])*0.5;
                let v2 = (verts[base_triangle_indices.x] + verts[base_triangle_indices.z])*0.5;
                let v3 = (verts[base_triangle_indices.y] + verts[base_triangle_indices.z])*0.5;

                let num_generated_vertices = verts.len();
                verts.extend_from_slice(&[v1,v2,v3]);


                let t1 = vek::Vec3::new(base_triangle_indices.x, num_generated_vertices, num_generated_vertices+1);
                let t2 = vek::Vec3::new(num_generated_vertices+1, num_generated_vertices, num_generated_vertices+2);
                let t3 = vek::Vec3::new(num_generated_vertices+2, num_generated_vertices, base_triangle_indices.y);
                let t4 = vek::Vec3::new(base_triangle_indices.z, num_generated_vertices+1, num_generated_vertices+2);
                next_triangles.extend_from_slice(&[t1,t2,t3,t4]);
            }

            previous_triangles.clear();
            previous_triangles.extend(next_triangles.iter());
        }

        tess_vert_count.push_str(&format!("{},", verts.len()));
        tess_tri_count.push_str(&format!("{},", next_triangles.len()));
        

        tess_verts_weights.push('{');
        for vertex in verts {
            tess_verts_weights.push_str(&format!(" float3({}, {}, {}), ", vertex.x, vertex.y, vertex.z));
        }
        tess_verts_weights.push_str("},");

        tess_tris.push('{');
        for tri in next_triangles {
            tess_tris.push_str(&format!(" uint3({}, {}, {}), ", tri.x, tri.y, tri.z));
        }
        tess_tris.push_str("},");
    }
    



    println!("{} }};", tess_vert_count);
    println!("{} }};", tess_tri_count);
    println!("{} }};", tess_verts_weights);
    println!("{} }};", tess_tris);
}