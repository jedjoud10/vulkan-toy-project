use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::renderer::GraphicsContext;

struct TesselationLevel {
    weights: Vec<vek::Vec3<f32>>,
    triangles: Vec<vek::Vec3<u32>>,
}



pub unsafe fn precompute_tesselation_buffer(
    ctx: &mut GraphicsContext,
) -> crate::buffer::Buffer {
    let depths = 4;
    let mut levels = Vec::<TesselationLevel>::new();

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

        levels.push(TesselationLevel {
            weights: verts,
            triangles: next_triangles.into_iter().map(|x| x.as_::<u32>()).collect::<_>(),
        });
    }

    #[derive(Clone, Copy, Pod, Zeroable, Debug)]
    #[repr(C)]
    struct Header {
        vertex_count: u32,
        triangle_count: u32,
        vertex_array_byte_offset: u32,
        triangle_array_byte_offset: u32,
    }
    
    let mut headers = Vec::<Header>::new();
    let headers_size = size_of::<Header>() * depths;

    let mut raw_bytes = Vec::<u8>::new();

    for level in levels {
        let vertex_bytes = bytemuck::cast_slice::<_, u8>(&level.weights);
        let triangle_bytes = bytemuck::cast_slice::<_, u8>(&level.triangles);
        

        headers.push(Header {
            vertex_count: level.weights.len() as u32,
            triangle_count: level.triangles.len() as u32,
            vertex_array_byte_offset: (raw_bytes.len() + headers_size) as u32,
            triangle_array_byte_offset: (raw_bytes.len() + headers_size + vertex_bytes.len()) as u32,
        });

        raw_bytes.extend_from_slice(vertex_bytes);
        raw_bytes.extend_from_slice(triangle_bytes);
    }

    //dbg!(&headers);


    let buffer = crate::buffer::create_buffer(ctx, raw_bytes.len() + headers_size as usize, "tesselation geometry buffer", vk::BufferUsageFlags::STORAGE_BUFFER);

    crate::buffer::write_to_buffer_with_offset(ctx, buffer.buffer, bytemuck::cast_slice(&headers), 0);
    crate::buffer::write_to_buffer_with_offset(ctx, buffer.buffer, &raw_bytes, headers_size as u64);
    
    
    buffer
}