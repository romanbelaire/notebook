For “engine-like” cross‑platform experiments with custom buttons/subwindows, a winit + wgpu style stack is a very good fit; you get a single GPU surface and treat every UI element as something you render and animate yourself.
​

Recommended architecture
Platform / windowing:

Use winit for cross‑platform windows, input events, and resize/fullscreen handling.
​

Rendering backend:

Use wgpu as your GPU abstraction (DX12/Metal/Vulkan/OpenGL behind the scenes) so you stay portable.
​

Your “UI”:

Implement your own scene graph: panels, buttons, and “subwindows” are quads with:

Position, size, z‑order.

Visual style (solid, gradient, glass, textured).

State (hovered, pressed, focused, etc.).

Each frame:

Poll winit events, update per-element state.

Run a small animation/physics step for positions, opacity, etc.

Issue wgpu draw calls for your elements.

This gives you game‑engine‑like control: you can slide, fling, stretch, or animate “windows” however you like, with deterministic timing.

Handling custom buttons and movement
Hit testing:

On mouse events, transform screen coordinates into your logical space and check which rectangles (buttons/subwindows) contain the point.

Maintain an “active” element for drag, press, etc.

Drag/move subwindows:

On mouse-down in a title bar region, capture that element and update its position each frame based on mouse delta.

Add spring/damping or inertial effects by integrating velocity instead of snapping directly.

Visual states:

Drive hover/press/disabled states via a small state machine per element and interpolate properties (color, blur amount, glow) instead of instant switches.

Because you own the render loop, you can keep per-frame latency very low and experiment with different interpolation/spring models without fighting a retained UI toolkit.
​

Adding “liquid glass” style effects
Render background content to an offscreen texture.

For a glass panel quad:

Sample the background texture with an offset based on a normal map or procedural noise to get refraction.

Mix in blur or approximate it with multi-sampling or a pre-blurred buffer depending on performance.

Add lighting/specular via a simple BRDF‑ish term or a gradient mask.

All of this is just a fragment shader on your panel geometry; with wgpu you can ship SPIR-V/WGSL shaders and tweak them easily.
​

Minimal “quality of life” helpers
Even if you go “engine style,” a couple of small utilities help:

A tiny layout helper (e.g., vertical/horizontal stacks and anchors) so you do not hand-position everything.

A small style system (struct of colors, radii, animation timings) that you can quickly tweak to explore different visual feels.

Optional: embed a debug overlay (FPS, frame time, element bounds) that you can toggle, similar to a game’s debug view.

Here is a minimal, engine-style cross‑platform skeleton using Rust + winit + wgpu, structured specifically for “moving subwindows” and custom buttons.

Project layout
text
your-app/
  Cargo.toml
  src/
    main.rs
    app.rs          // App struct, update() + render()
    ui/
      mod.rs
      window.rs     // Subwindow (panel) abstraction
      button.rs     // Custom button abstraction
    gfx/
      mod.rs
      renderer.rs   // Wgpu setup, pipelines, shaders
      types.rs      // Vertex, uniforms, etc.
Core concepts
App: owns the scene graph (subwindows, buttons) and global state.

SubWindow: movable panel with position/size, children, and velocity.

Button: hit-testable region with state (normal/hover/pressed) and callbacks.

Renderer: draws quads for subwindows/buttons; later you plug liquid-glass shaders here.

Cargo.toml (key parts)
text
[package]
name = "ui_engine"
version = "0.1.0"
edition = "2021"

[dependencies]
winit = "0.29"
wgpu = "0.20"
pollster = "0.3"
glam = "0.27"    # for Vec2/Mat4, etc.
src/main.rs – entry + event loop
rust
mod app;
mod gfx;
mod ui;

use crate::app::App;
use crate::gfx::renderer::Renderer;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("UI engine experiment")
        .build(&event_loop)
        .unwrap();

    // WGPU renderer (device, queue, swapchain, pipelines)
    let mut renderer = pollster::block_on(Renderer::new(&window));

    // App state (subwindows, buttons, etc.)
    let mut app = App::new(window.inner_size().into());

    let mut last_frame = std::time::Instant::now();

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => {
                        renderer.resize(size);
                        app.resize(size.into());
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        app.on_mouse_button(button, state);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        app.on_cursor_moved(position);
                    }
                    WindowEvent::KeyboardInput { input, .. } => {
                        app.on_keyboard(input);
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    // Per-frame update + render
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(last_frame).as_secs_f32();
                    last_frame = now;

                    app.update(dt);
                    renderer.render(&app).unwrap();
                }
                _ => {}
            }
        })
        .unwrap();
}
src/app.rs – app state and logic
rust
use crate::ui::{SubWindow, Button};
use glam::Vec2;
use winit::event::{ElementState, MouseButton, KeyboardInput};

pub struct App {
    pub windows: Vec<SubWindow>,
    pub mouse_pos: Vec2,
    dragging_id: Option<usize>,
    drag_offset: Vec2,
    pub viewport_size: Vec2,
}

impl App {
    pub fn new(viewport_size: (u32, u32)) -> Self {
        let viewport = Vec2::new(viewport_size.0 as f32, viewport_size.1 as f32);

        let mut win = SubWindow::new(Vec2::new(100.0, 100.0), Vec2::new(300.0, 200.0));
        win.add_button(Button::new(Vec2::new(20.0, 40.0), Vec2::new(80.0, 30.0), "OK"));

        Self {
            windows: vec![win],
            mouse_pos: Vec2::ZERO,
            dragging_id: None,
            drag_offset: Vec2::ZERO,
            viewport_size: viewport,
        }
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        self.viewport_size = Vec2::new(size.0 as f32, size.1 as f32);
    }

    pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button != MouseButton::Left {
            return;
        }

        match state {
            ElementState::Pressed => {
                // hit test windows top to bottom (front to back)
                for (id, win) in self.windows.iter_mut().enumerate().rev() {
                    if win.hit_title_bar(self.mouse_pos) {
                        self.dragging_id = Some(id);
                        self.drag_offset = self.mouse_pos - win.position;
                        // bring to front
                        let w = self.windows.remove(id);
                        self.windows.push(w);
                        self.dragging_id = Some(self.windows.len() - 1);
                        return;
                    }

                    if win.hit_any_button(self.mouse_pos) {
                        win.on_mouse_down(self.mouse_pos);
                        return;
                    }
                }
            }
            ElementState::Released => {
                if let Some(id) = self.dragging_id.take() {
                    self.windows[id].velocity = Vec2::ZERO; // or keep for inertia
                }
                for win in &mut self.windows {
                    win.on_mouse_up(self.mouse_pos);
                }
            }
        }
    }

    pub fn on_cursor_moved(&mut self, pos: winit::dpi::PhysicalPosition<f64>) {
        self.mouse_pos = Vec2::new(pos.x as f32, pos.y as f32);

        if let Some(id) = self.dragging_id {
            let target = self.mouse_pos - self.drag_offset;
            self.windows[id].position = target;
        } else {
            for win in &mut self.windows {
                win.on_mouse_move(self.mouse_pos);
            }
        }
    }

    pub fn on_keyboard(&mut self, _input: KeyboardInput) {
        // for future shortcuts
    }

    pub fn update(&mut self, dt: f32) {
        for win in &mut self.windows {
            win.update(dt, self.viewport_size);
        }
    }
}
src/ui/window.rs – subwindow abstraction
rust
use crate::ui::Button;
use glam::Vec2;

pub struct SubWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub velocity: Vec2,
    pub buttons: Vec<Button>,
}

impl SubWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        Self {
            position,
            size,
            velocity: Vec2::ZERO,
            buttons: Vec::new(),
        }
    }

    pub fn add_button(&mut self, button: Button) {
        self.buttons.push(button);
    }

    pub fn hit_title_bar(&self, p: Vec2) -> bool {
        let rel = p - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= 30.0
    }

    pub fn hit_any_button(&self, p: Vec2) -> bool {
        let rel = p - self.position;
        self.buttons.iter().any(|b| b.contains(rel))
    }

    pub fn on_mouse_down(&mut self, p: Vec2) {
        let rel = p - self.position;
        for b in &mut self.buttons {
            if b.contains(rel) {
                b.on_press();
            }
        }
    }

    pub fn on_mouse_up(&mut self, p: Vec2) {
        let rel = p - self.position;
        for b in &mut self.buttons {
            if b.contains(rel) {
                b.on_release();
            } else {
                b.on_cancel();
            }
        }
    }

    pub fn on_mouse_move(&mut self, p: Vec2) {
        let rel = p - self.position;
        for b in &mut self.buttons {
            b.on_hover(rel);
        }
    }

    pub fn update(&mut self, dt: f32, viewport: Vec2) {
        // optional inertia / physics
        self.position += self.velocity * dt;

        // simple bounds clamp
        if self.position.x < 0.0 {
            self.position.x = 0.0;
            self.velocity.x = 0.0;
        }
        if self.position.y < 0.0 {
            self.position.y = 0.0;
            self.velocity.y = 0.0;
        }
        if self.position.x + self.size.x > viewport.x {
            self.position.x = viewport.x - self.size.x;
            self.velocity.x = 0.0;
        }
        if self.position.y + self.size.y > viewport.y {
            self.position.y = viewport.y - self.size.y;
            self.velocity.y = 0.0;
        }
    }
}
src/ui/button.rs – custom button state
rust
use glam::Vec2;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ButtonState {
    Normal,
    Hover,
    Pressed,
}

pub struct Button {
    pub position: Vec2,
    pub size: Vec2,
    pub label: String,
    pub state: ButtonState,
}

impl Button {
    pub fn new(position: Vec2, size: Vec2, label: &str) -> Self {
        Self {
            position,
            size,
            label: label.to_string(),
            state: ButtonState::Normal,
        }
    }

    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.position.x
            && p.x <= self.position.x + self.size.x
            && p.y >= self.position.y
            && p.y <= self.position.y + self.size.y
    }

    pub fn on_press(&mut self) {
        self.state = ButtonState::Pressed;
    }

    pub fn on_release(&mut self) {
        // in a real system you’d fire a callback here
        self.state = ButtonState::Hover;
    }

    pub fn on_cancel(&mut self) {
        self.state = ButtonState::Normal;
    }

    pub fn on_hover(&mut self, p: Vec2) {
        if self.contains(p) {
            if self.state == ButtonState::Normal {
                self.state = ButtonState::Hover;
            }
        } else if self.state != ButtonState::Pressed {
            self.state = ButtonState::Normal;
        }
    }
}
src/gfx/renderer.rs – wgpu setup + simple quad rendering
This is the piece you’ll extend with shaders for liquid glass. Minimal version:

rust
use crate::app::App;
use crate::ui::ButtonState;
use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;
use winit::window::Window;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    // vertex/index buffers, etc.
}

impl Renderer {
    pub async fn new(window: &Window) -> Self {
        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface(window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo, // vsync
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // TODO: create pipeline with a WGSL shader that draws colored quads.
        let pipeline = Self::create_pipeline(&device, config.format);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn render(&mut self, app: &App) -> anyhow::Result<()> {
        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| anyhow::anyhow!("get_current_texture: {:?}", e))?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.pipeline);

            // For each subwindow and its buttons, issue quads with different colors
            // depending on ButtonState; later this becomes your glass shader.
            for win in &app.windows {
                // draw window background quad
                // draw title bar quad
                // draw buttons
                for b in &win.buttons {
                    let color = match b.state {
                        ButtonState::Normal => [0.2, 0.2, 0.25, 1.0],
                        ButtonState::Hover => [0.3, 0.3, 0.4, 1.0],
                        ButtonState::Pressed => [0.1, 0.6, 0.9, 1.0],
                    };
                    // push Rect{position = win.position + b.position, size = b.size, color}
                    // into a dynamic vertex buffer or uniform for instanced rendering
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn create_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        // Very simple pipeline: position + color, no textures yet.
        // Replace shader later with liquid-glass WGSL.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ui_shader.wgsl").into()),
        });

        // ... create pipeline layout, vertex buffers, etc.
        // Left schematic; you can fill this using any wgpu example as reference.

        unimplemented!()
    }
}
You can lift boilerplate from the official wgpu examples to finish create_pipeline and the vertex layout.

How to plug in “liquid glass” and more
Once this skeleton runs and you see movable gray panels and color-changing buttons:

Add an offscreen color buffer for background content.

In ui_shader.wgsl, sample that background texture with a distortion field to create refraction.

Drive distortion parameters from App state (e.g., “subwindow focus”, hover), passed as uniforms.

Experiment with spring/inertia in SubWindow::update for more “liquid” movement.