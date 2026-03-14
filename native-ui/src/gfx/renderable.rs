use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use crate::app::App;

/// Trait for UI components that can be rendered
/// Each component is responsible for generating its own vertices
pub trait Renderable {
    /// Render this component and append vertices to the provided vector
    /// Components can also queue text through the renderer
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>);
    
    /// Z-order for this component (lower values render first/behind)
    /// Default layers:
    /// - 0: Background elements
    /// - 10: Sidebar, main content areas
    /// - 20: Glow effects
    /// - 30: Modals, dialogs
    /// - 100: Header (always on top)
    fn z_order(&self) -> i32;
}

/// Helper struct to wrap renderables with their z-order for sorting
pub struct RenderableComponent {
    pub z_order: i32,
    pub name: &'static str,
    pub render_fn: Box<dyn Fn(&mut Renderer, &mut App, &mut Vec<Vertex>)>,
}

impl RenderableComponent {
    pub fn new<F>(name: &'static str, z_order: i32, render_fn: F) -> Self
    where
        F: Fn(&mut Renderer, &mut App, &mut Vec<Vertex>) + 'static,
    {
        Self {
            z_order,
            name,
            render_fn: Box::new(render_fn),
        }
    }
}

