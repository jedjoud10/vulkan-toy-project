use std::time::{Duration, Instant};
use std::fmt::Write;
use ash::vk;

use crate::per_frame_data;

pub const NUM_TIMESTAMP_QUERIES: usize = 4;
pub const BUFFER_SIZE: usize = 8;

struct SingleRegion {
    name: &'static str,
    start_query_index: usize,
    end_query_index: usize,
    buffer: [f64; BUFFER_SIZE]
}

impl SingleRegion {
    pub fn new(name: &'static str, start_query_index: usize, end_query_index: usize) -> Self {
        Self {
            name,
            start_query_index,
            end_query_index,
            buffer: Default::default(),
        }
    }

    pub fn push(&mut self, delta_in_ms: f64) {
        self.buffer.rotate_right(1);
        self.buffer[0] = delta_in_ms;
    }

    pub fn get_average_in_ms(&self) -> f64 {
        self.buffer.iter().sum::<f64>() / BUFFER_SIZE as f64
    }
}

pub struct QueryPoolStatistics {
    regions: [SingleRegion; 4]
}

impl QueryPoolStatistics {
    pub fn new() -> Self {
        let entire_frame = SingleRegion::new("entire frame", 0, 3);
        let skybox_region = SingleRegion::new("skybox pass", 0, 1);
        let compute_region = SingleRegion::new("main frame pass", 1, 2);
        let bloom_region = SingleRegion::new("postprocess pass", 2, 3);
    
        Self {
            regions: [entire_frame, skybox_region, compute_region, bloom_region]
        }
    }

    pub unsafe fn import_data(&mut self, frame_count: u64, device: &ash::Device, query_pool: vk::QueryPool, timestamp_period: f32) {
        // wait for a few frames so that the queries get populated
        if frame_count <= per_frame_data::FRAMES_IN_FLIGHT as u64 {
            return;
        }


        // try to fetch timestamp queries
        let mut timestamps = [0u64; NUM_TIMESTAMP_QUERIES];
        let okay = device.get_query_pool_results(query_pool, 0, &mut timestamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).is_ok();

        if okay {
            for region in self.regions.iter_mut() {
                let start_timestamp_value = timestamps[region.start_query_index];
                let end_timestamp_value = timestamps[region.end_query_index];                

                let delta_in_ms = ((end_timestamp_value.saturating_sub(start_timestamp_value)) as f64 * timestamp_period as f64) / 1000000.0f64;
                region.push(delta_in_ms);
            }
        }
    }

    pub fn add_to_debug_text(&self, debug_text: &mut crate::debug_text::DebugText) {

        for region in self.regions.iter() {
            writeln!(debug_text, "Query Region \" {} \": {:.2}ms", region.name, region.get_average_in_ms()).unwrap();
        }
    }

    pub fn get_compute_region_duration(&self) -> f64 {
        self.regions[2].get_average_in_ms()
    }
}

pub unsafe fn create_query_pool(
    device: &ash::Device
) -> vk::QueryPool {
    let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(NUM_TIMESTAMP_QUERIES as u32);
    let query = device.create_query_pool(&create_info, None).unwrap();
    device.reset_query_pool(query, 0, NUM_TIMESTAMP_QUERIES as u32);
    query
}