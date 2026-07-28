use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::Allocator;

use crate::{buffer, renderer::GraphicsContext};

const DEBUG_TEXT_BUFFER_SIZE_BYTES: usize = 1024;

pub struct DebugText {
    pub buffer: buffer::Buffer,
    pub text: String,
}

impl DebugText {
    pub unsafe fn new(ctx: &mut GraphicsContext) -> Self {
        Self {
            buffer: buffer::create_buffer(ctx, DEBUG_TEXT_BUFFER_SIZE_BYTES, "debug text", vk::BufferUsageFlags::STORAGE_BUFFER),
            text: String::new()
        }
    }
    
    pub unsafe fn update_debug_text(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let bytes = convert_string_to_debug_text_bytes(&self.text);
        dbg!(bytes.len());
        device.cmd_update_buffer(cmd, self.buffer.buffer, 0, &bytes);
        self.text.clear();
    }
    
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.buffer.destroy(device, allocator);
    }
}


impl std::fmt::Write for DebugText {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.text.write_str(s)
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct DebugTextLineHeader {
    start_byte_offset: u32,
    char_count: u32,
}

fn convert_string_to_debug_text_bytes(text: &str) -> Vec<u8> {
    // write total number of lines
    let total_num_lines = text.lines().count() as u32;
    let mut bytes = bytemuck::bytes_of(&total_num_lines).to_vec();

    // write headers for each line
    let mut prefix_sum_chars_only = 0u32;
    for line in text.lines() {
        // calculate the total size in bytes prior to the actual text data
        let mut total_size_prior = total_num_lines * size_of::<DebugTextLineHeader>() as u32;

        // plus also the u32 to indicate the line count
        total_size_prior += size_of::<u32>() as u32;

        let header_for_line = DebugTextLineHeader {
            start_byte_offset: total_size_prior + prefix_sum_chars_only, 
            char_count: line.as_bytes().len() as u32,
        };
        
        bytes.extend_from_slice(bytemuck::bytes_of(&header_for_line));
        prefix_sum_chars_only += line.as_bytes().len() as u32;
    }

    for line in text.lines() {
        bytes.extend_from_slice(line.as_bytes());
    }

    // make sure bytes is word aligned 
    bytes.resize(bytes.len().div_ceil(4) * 4, 0);

    bytes
}