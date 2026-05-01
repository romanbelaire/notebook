/// Window component wrappers that implement Renderable
/// These components wrap window data and render it as part of the component hierarchy
use glam::Vec2;
use crate::ui::core::Rect;
use crate::ui::shadow::ShadowSpec;
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use crate::app::App;
use crate::ui::components::Renderable;
use crate::gfx::renderer::CompositeLayer;

/// Header component - wraps HeaderWindow
pub struct HeaderComponent {
    component_id: String,
    pub shadow: Option<ShadowSpec>,
}

impl HeaderComponent {
    pub fn new() -> Self {
        Self {
            component_id: "header".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for HeaderComponent {
    fn z_order(&self) -> i32 {
        100
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "HeaderComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::header::HEADER_VIEWPORT.render(renderer, app, vertices, dirty_rect);
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
    pub shadow: Option<ShadowSpec>,
}

impl SidebarComponent {
    pub fn new() -> Self {
        Self {
            component_id: "sidebar".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for SidebarComponent {
    fn z_order(&self) -> i32 {
        20
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("root"), "SidebarComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::sidebar::SIDEBAR_VIEWPORT.render(renderer, app, vertices, dirty_rect);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        Some(crate::gfx::components::sidebar::sidebar_chrome_rect(app))
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
    pub shadow: Option<ShadowSpec>,
}

impl SidebarContentComponent {
    pub fn new() -> Self {
        Self {
            component_id: "sidebar_content".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for SidebarContentComponent {
    fn z_order(&self) -> i32 {
        20
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("sidebar"), "SidebarContentComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::sidebar_content::SIDEBAR_CONTENT_VIEWPORT.render(renderer, app, vertices, dirty_rect);
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
    pub shadow: Option<ShadowSpec>,
}

impl ChatComponent {
    pub fn new() -> Self {
        Self {
            component_id: "chat".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for ChatComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.set_composite_layer(CompositeLayer::MainContent);
        renderer.validate_component(&self.component_id, Some("root"), "ChatComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::chat::CHAT_VIEWPORT.render(renderer, app, vertices, dirty_rect);
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
    pub shadow: Option<ShadowSpec>,
}

impl LibraryComponent {
    pub fn new() -> Self {
        Self {
            component_id: "library".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for LibraryComponent {
    fn z_order(&self) -> i32 {
        11
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.set_composite_layer(CompositeLayer::MainContent);
        renderer.validate_component(&self.component_id, Some("root"), "LibraryComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::library::LIBRARY_VIEWPORT.render(renderer, app, vertices, dirty_rect);
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
    pub shadow: Option<ShadowSpec>,
}

impl DataComponent {
    pub fn new() -> Self {
        Self {
            component_id: "data".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for DataComponent {
    fn z_order(&self) -> i32 {
        12
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.set_composite_layer(CompositeLayer::MainContent);
        renderer.validate_component(&self.component_id, Some("root"), "DataComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::data::DATA_VIEWPORT.render(renderer, app, vertices, dirty_rect);
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
    pub shadow: Option<ShadowSpec>,
}

impl SettingsComponent {
    pub fn new() -> Self {
        Self {
            component_id: "settings".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for SettingsComponent {
    fn z_order(&self) -> i32 {
        13
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.set_composite_layer(CompositeLayer::MainContent);
        renderer.validate_component(&self.component_id, Some("root"), "SettingsComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
        renderer.push_parent(self.component_id.clone());
        crate::gfx::components::settings::SETTINGS_VIEWPORT.render(renderer, app, vertices, dirty_rect);
        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.settings_window
            .borrow()
            .as_ref()
            .map(|c| Rect::new(c.position.x, c.position.y, c.size.x, c.size.y))
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
    pub shadow: Option<ShadowSpec>,
}

impl NotepadComponent {
    pub fn new() -> Self {
        Self {
            component_id: "notepad".to_string(),
            shadow: None,
        }
    }

    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for NotepadComponent {
    fn z_order(&self) -> i32 {
        14
    }

    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.set_composite_layer(CompositeLayer::MainContent);
        renderer.validate_component(&self.component_id, Some("root"), "NotepadComponent");
        if let Some(spec) = &self.shadow {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), spec);
            }
        }
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

