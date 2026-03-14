/// Window component wrappers that implement Renderable
/// These components wrap window data and render it as part of the component hierarchy
use glam::Vec2;
use crate::ui::core::Rect;
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use crate::app::App;
use crate::ui::components::Renderable;

/// Header component - wraps HeaderWindow
pub struct HeaderComponent {
    component_id: String,
}

impl HeaderComponent {
    pub fn new() -> Self {
        Self {
            component_id: "header".to_string(),
        }
    }
}

impl Renderable for HeaderComponent {
    fn z_order(&self) -> i32 {
        100
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "HeaderComponent");
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::header::render_header(renderer, app, vertices);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        Some(Rect::new(
            app.header.position.x,
            app.header.position.y,
            app.header.size.x,
            app.header.size.y,
        ))
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by HeaderWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 60.0)
    }
}

/// Sidebar component - wraps SidebarWindow
pub struct SidebarComponent {
    component_id: String,
}

impl SidebarComponent {
    pub fn new() -> Self {
        Self {
            component_id: "sidebar".to_string(),
        }
    }
}

impl Renderable for SidebarComponent {
    fn z_order(&self) -> i32 {
        20
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "SidebarComponent");
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::sidebar::render_sidebar(renderer, app, vertices);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        let y = app.header.size.y;
        let h = app.viewport_size.y - y;
        Some(Rect::new(0.0, y, app.sidebar.current_width, h))
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by SidebarWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Sidebar content component - wraps sidebar content rendering
pub struct SidebarContentComponent {
    component_id: String,
}

impl SidebarContentComponent {
    pub fn new() -> Self {
        Self {
            component_id: "sidebar_content".to_string(),
        }
    }
}

impl Renderable for SidebarContentComponent {
    fn z_order(&self) -> i32 {
        20
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("sidebar"), "SidebarContentComponent");
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::sidebar_content::render_sidebar_content(renderer, app, vertices);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by SidebarWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Chat component - wraps ChatWindow
pub struct ChatComponent {
    component_id: String,
}

impl ChatComponent {
    pub fn new() -> Self {
        Self {
            component_id: "chat".to_string(),
        }
    }
}

impl Renderable for ChatComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "ChatComponent");
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::chat::render_chat_window(renderer, app, vertices);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.chat_window.as_ref().map(|c| Rect::new(c.position.x, c.position.y, c.size.x, c.size.y))
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by ChatWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Library component - wraps LibraryWindow
pub struct LibraryComponent {
    component_id: String,
}

impl LibraryComponent {
    pub fn new() -> Self {
        Self {
            component_id: "library".to_string(),
        }
    }
}

impl Renderable for LibraryComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "LibraryComponent");
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::library::render_library(renderer, app, vertices);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.library_window.as_ref().map(|c| Rect::new(c.position.x, c.position.y, c.size.x, c.size.y))
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by LibraryWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Data component - wraps IngestWindow
pub struct DataComponent {
    component_id: String,
}

impl DataComponent {
    pub fn new() -> Self {
        Self {
            component_id: "data".to_string(),
        }
    }
}

impl Renderable for DataComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "DataComponent");
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::data::render_data(renderer, app, vertices);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.ingest_window.as_ref().map(|c| Rect::new(c.position.x, c.position.y, c.size.x, c.size.y))
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by IngestWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Settings component - wraps SettingsWindow
pub struct SettingsComponent {
    component_id: String,
}

impl SettingsComponent {
    pub fn new() -> Self {
        Self {
            component_id: "settings".to_string(),
        }
    }
}

impl Renderable for SettingsComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "SettingsComponent");
        renderer.push_parent(self.component_id.clone());
        unsafe {
            let app_ptr: *mut App = std::ptr::addr_of!(*app).cast_mut();
            let app_mut = &mut *app_ptr;
            crate::gfx::components::settings::render_settings(renderer, app_mut, vertices);
        }
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.settings_window.as_ref().map(|c| Rect::new(c.position.x, c.position.y, c.size.x, c.size.y))
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by SettingsWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Notepad component - wraps NotepadWindow
pub struct NotepadComponent {
    component_id: String,
}

impl NotepadComponent {
    pub fn new() -> Self {
        Self {
            component_id: "notepad".to_string(),
        }
    }
}

impl Renderable for NotepadComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "NotepadComponent");
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::notepad::render_notepad(renderer, app, vertices);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.notepad_window.as_ref().map(|c| Rect::new(c.position.x, c.position.y, c.size.x, c.size.y))
    }

    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        // Layout is managed by NotepadWindow itself
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

