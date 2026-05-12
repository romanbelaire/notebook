#![allow(warnings)]
mod app;
mod clipboard;
mod gfx;
mod ui;
mod state;
mod api;
mod utils;
mod stylus;
mod persistence;
mod knowledge;


use crate::app::App;
use crate::gfx::renderer::Renderer;
use winit::{
    event::*,
    event_loop::{EventLoop, ActiveEventLoop},
    window::Window,
    application::ApplicationHandler,
    keyboard::PhysicalKey,
};
use crate::app::WindowControlEvent;
use std::sync::Arc;

struct AppState {
    window: Arc<Window>,
    renderer: Renderer,
    app: App,
    is_maximized: bool,
    last_frame: std::time::Instant,
}

struct NotebookApp {
    state: Option<AppState>,
    proxy: Option<winit::event_loop::EventLoopProxy<WindowControlEvent>>,
}

impl ApplicationHandler<WindowControlEvent> for NotebookApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("Notebook - Native UI")
                .with_inner_size(winit::dpi::LogicalSize::new(1200, 800))
                .with_decorations(false);
            
            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
            let renderer = pollster::block_on(Renderer::new(window.clone()));
            let mut app = App::new(window.inner_size().into());
            
            // Use the proxy we saved from main()
            if let Some(proxy) = &self.proxy {
                app.set_window_proxy(proxy.clone());
            }
            
            self.state = Some(AppState {
                window,
                renderer,
                app,
                is_maximized: false,
                last_frame: std::time::Instant::now(),
            });
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, control_event: WindowControlEvent) {
        if let Some(state) = &mut self.state {
            match control_event {
                WindowControlEvent::Minimize => {
                    state.window.set_minimized(true);
                }
                WindowControlEvent::ToggleMaximize => {
                    state.is_maximized = !state.is_maximized;
                    state.window.set_maximized(state.is_maximized);
                }
                WindowControlEvent::Close => {
                    event_loop.exit();
                }
                WindowControlEvent::DragWindow => {
                    let _ = state.window.drag_window();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: winit::window::WindowId, event: WindowEvent) {
        if let Some(state) = &mut self.state {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    state.renderer.resize(size);
                    state.app.resize(size.into());
                }
                WindowEvent::MouseInput { state: button_state, button, .. } => {
                    state.app.on_mouse_button(button, button_state);
                    state.window.request_redraw();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    state.app.on_cursor_moved(position);
                    state.window.set_cursor_icon(state.app.desired_cursor_icon);
                    state.window.request_redraw();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    // Handle keyboard shortcuts and navigation
                    state.app.on_keyboard(&event);
                    
                    // Track key press/release
                    if let PhysicalKey::Code(key_code) = event.physical_key {
                        if event.state == ElementState::Pressed {
                            state.app.pressed_keys.insert(key_code);
                        } else {
                            state.app.on_key_released(key_code);
                            state.app.pressed_keys.remove(&key_code);
                        }
                    }
                    
                    // Handle text input from keyboard
                    if event.state == ElementState::Pressed {
                        // Handle spacebar explicitly (may not always generate Character event)
                        if let PhysicalKey::Code(winit::keyboard::KeyCode::Space) = event.physical_key {
                            // Only send space if no modifiers are pressed (to avoid interfering with shortcuts)
                            if !state.app.modifiers.intersects(
                                winit::keyboard::ModifiersState::CONTROL 
                                | winit::keyboard::ModifiersState::ALT 
                                | winit::keyboard::ModifiersState::SUPER
                            ) {
                                state.app.on_char_received(' ');
                            }
                        } else if let winit::keyboard::Key::Character(text) = &event.logical_key {
                            for ch in text.chars() {
                                if !ch.is_control() && ch != '\u{7f}' {
                                    state.app.on_char_received(ch);
                                }
                            }
                        }
                    }
                    state.window.request_redraw();
                }
                WindowEvent::ModifiersChanged(modifiers) => {
                    state.app.modifiers = modifiers.state();
                }
                WindowEvent::Ime(ime) => {
                    match ime {
                        winit::event::Ime::Commit(text) => {
                            for ch in text.chars() {
                                // Filter out control characters
                                if !ch.is_control() && ch != '\u{7f}' {
                                    state.app.on_char_received(ch);
                                }
                            }
                            state.window.request_redraw();
                        }
                        winit::event::Ime::Preedit(_, _) => {}
                        winit::event::Ime::Enabled => {}
                        winit::event::Ime::Disabled => {}
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    state.app.on_mouse_wheel(delta);
                    state.window.request_redraw();
                }
                WindowEvent::Focused(focused) => {
                    if focused {
                        state.app.on_window_focus();
                    } else {
                        state.app.on_window_blur();
                    }
                    state.window.request_redraw();
                }
                WindowEvent::Moved(position) => {
                    state.app.on_window_moved(position);
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    state.app.on_scale_factor_changed(scale_factor);
                    state.window.request_redraw();
                }
                WindowEvent::DroppedFile(path) => {
                    state.app.on_file_drop(vec![path], state.app.mouse_pos);
                    state.window.request_redraw();
                }
                WindowEvent::HoveredFile(path) => {
                    state.app.on_file_hover(vec![path], state.app.mouse_pos);
                    state.window.request_redraw();
                }
                WindowEvent::HoveredFileCancelled => {
                    state.app.on_file_hover_cancelled();
                    state.window.request_redraw();
                }
                WindowEvent::Touch(touch) => {
                    state.app.on_touch(&touch);
                    state.window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(state.last_frame).as_secs_f32();
                    state.last_frame = now;

                    state.app.update(dt);
                    state.app.check_api_responses();
                    state.app.check_graph_loaded();
                    state.app.check_graph_send_responses();
                    state.app.check_collections_responses();
                    state.app.check_context_pool_responses();
                    state.app.check_papers_responses();
                    state.app.check_insights_responses();
                    state.app.check_ingest_responses();
                    state.app.check_task_status_responses();
                    state.app.check_pdf_responses();
                    state.app.check_note_content_responses();
                    
                    if let Err(e) = state.renderer.render(&mut state.app) {
                        eprintln!("Render error: {:?}", e);
                    }
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Only request redraw when something is animating (physics, springs, lerps).
        // When fully idle (e.g. graph at rest), skip redraws to reduce CPU/GPU and fix lag.
        if let Some(state) = &self.state {
            if state.app.needs_continuous_redraw() {
                state.window.request_redraw();
            }
        }
    }
}

fn main() {
    env_logger::init();

    // Create Tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _enter = rt.enter();

    let event_loop = EventLoop::<WindowControlEvent>::with_user_event().build().unwrap();
    let proxy = event_loop.create_proxy();
    let mut app = NotebookApp { 
        state: None,
        proxy: Some(proxy),
    };
    
    event_loop.run_app(&mut app).unwrap();
}
