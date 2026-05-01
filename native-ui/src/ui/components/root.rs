/// Root component - the only component that can be instantiated without a parent
/// All other components must be created through their parent component
use glam::Vec2;
use crate::ui::core::Rect;
use crate::ui::shadow::ShadowSpec;
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use super::Renderable;

/// Root component - the base of the component hierarchy
/// This is the only component that can be created without a parent
pub struct Root {
    pub children: Vec<Box<dyn Renderable>>,
    pub rect: Rect,
    /// Last app.layout_generation we ran update_layout for; skip layout when unchanged.
    pub last_layout_generation: u64,
    pub shadow: Option<ShadowSpec>,
}

impl Root {
    /// Create a new Root component (only one should exist)
    /// This is the only component that can be created without a parent
    pub fn new(viewport_size: Vec2) -> Self {
        Self {
            children: Vec::new(),
            rect: Rect::new(0.0, 0.0, viewport_size.x, viewport_size.y),
            last_layout_generation: u64::MAX, // First frame always runs layout
            shadow: None,
        }
    }
    
    /// Add a child component to the root
    /// All components must be added through their parent
    pub fn add_child(&mut self, child: Box<dyn Renderable>) {
        self.children.push(child);
    }

    /// Attach a drop shadow (rare — typically Root has no parent to cast onto).
    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

use crate::app::App;

impl Renderable for Root {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        if !renderer.should_skip_component("root") {
            renderer.validate_component("root", None, "Root");
        }
        if let Some(spec) = &self.shadow {
            renderer.queue_shadow(&self.rect, 0.0, spec);
        }
        renderer.push_parent("root".to_string());
        let mut sorted: Vec<_> = self.children.iter().enumerate().collect();
        sorted.sort_by_key(|(i, c)| (c.z_order(), *i));
        for (_, child) in sorted {
            let bounds = child.bounds_from_app(app).unwrap_or_else(|| child.bounds());
            let in_dirty = dirty_rect.map(|d| bounds.intersects(&d)).unwrap_or(true);
            if !in_dirty {
                continue;
            }
            child.render(renderer, app, vertices, dirty_rect);
            // Flush after each child so RenderBatch.layer matches each component's set_composite_layer.
            // Batching all z=20 children into one add_vertices used the last child's layer (SidebarChrome),
            // skipping MainContent quads in pass 1 and drawing them in pass 2 after MainContent Vello text.
            if !vertices.is_empty() {
                renderer.add_vertices(vertices, None);
                vertices.clear();
            }
        }
        renderer.pop_parent();
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, dirty_rect: Option<Rect>, app: Option<&App>) {
        self.rect = available_rect;
        let app = match app {
            Some(a) => a,
            None => {
                for child in &mut self.children {
                    child.update_layout(available_rect, dirty_rect, None);
                }
                return;
            }
        };
        for child in &mut self.children {
            let bounds = child.bounds_from_app(app).unwrap_or_else(|| child.bounds());
            let in_dirty = dirty_rect.map(|d| bounds.intersects(&d)).unwrap_or(true);
            if in_dirty {
                child.update_layout(available_rect, dirty_rect, Some(app));
            }
        }
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

