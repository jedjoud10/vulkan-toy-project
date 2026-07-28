use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Statistics {
    delta_ms_buffer: [f64; 8],
}

// TODO: rewrite this to an actual stats performance metric thingy
impl Statistics {
    pub fn push_query_timings(&mut self, delta_in_ms: f64) {
        self.delta_ms_buffer.rotate_right(1);
        self.delta_ms_buffer[0] = delta_in_ms;
    }

    pub fn end_of_frame(&mut self, frame: u64) {
    }

    pub fn start_benchmarking(&mut self, frame: u64) {
    }

    pub fn end_benchmarking(&mut self) {
    }

    pub fn get_average_in_ms(&self) -> f64 {
        self.delta_ms_buffer.iter().sum::<f64>() / self.delta_ms_buffer.len() as f64
    }
}