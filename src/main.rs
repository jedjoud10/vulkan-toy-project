#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused)]


mod debug;
mod device;
mod input;
mod instance;
mod movement;
mod physical_device;
mod pipeline;
mod swapchain;
mod ticker;
mod buffer;
mod statistics;
mod utils;
mod renderer;
mod skybox;
mod others;
mod render_targets_data;
mod per_frame_data;
mod samplers;
mod tesselation;
mod voxel;

use clap::Parser;
use std::ops::ControlFlow;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowId;
use renderer::InternalApp;


#[derive(clap::Parser, Debug)]
#[command(about = "Vulkan Experiments", long_about = None)]
struct Args {
    /// Factor to use to decrease the screen resolution
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=4))]
    downscale_factor: u32,

    /// Setting to start in fullscreen from the start. This can be toggled in-game using F5
    #[arg(long, default_value_t = false)]
    fullscreen: bool,

    /// Enable validation layers and debug stuff even when debug_assertions are disabled
    #[arg(long, default_value_t = false)]
    enable_debug_stuff: bool,
}

struct App {
    internal: Option<InternalApp>,
    args: Option<Args>,
    start: Instant,
    last: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        unsafe {
            self.internal = Some(InternalApp::new(event_loop, self.args.take().unwrap()));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => unsafe {
                event_loop.exit();
                self.internal.take().unwrap().destroy();
            },
            WindowEvent::RedrawRequested => unsafe {
                let inner = self.internal.as_mut().unwrap();
                let new = Instant::now();
                let elapsed = (new - self.start).as_secs_f32();
                let delta = (new - self.last).as_secs_f32();

                if let ControlFlow::Break(_) = inner.pre_render(delta) {
                    event_loop.exit();
                    self.internal.take().unwrap().destroy();
                    return;
                }

                inner.window.request_redraw();
                inner.render(delta, elapsed);
                self.last = new;
                input::update(&mut inner.input);
            },
            WindowEvent::Resized(_) => {
                let inner = self.internal.as_mut().unwrap();
                inner.was_resized = true;
            },

            // This is horrid...
            _ => {
                let inner = self.internal.as_mut().unwrap();
                input::window_event(&mut inner.input, &event);
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let inner = self.internal.as_mut().unwrap();
        input::device_event(&mut inner.input, &event);
    }
}

pub fn main() {
    let args = Args::parse();
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .parse_default_env()
        .format_timestamp_millis()
        .format_file(true)
        .format_line_number(true)
        .init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        start: Instant::now(),
        last: Instant::now(),
        internal: None,
        args: Some(args),
    };
    event_loop.run_app(&mut app).unwrap();
}
