/// Component-based UI system
/// 
/// This module implements a composable, modular UI architecture where every
/// UI element is a self-contained Renderable that can be nested and composed.
/// 
/// Architecture: All components must be created through their parent component.
/// Only the Root component can be instantiated without a parent.
pub mod sidebar;
pub mod root;
pub mod window_components;
pub mod list;

use glam::Vec2;
use crate::ui::core::Rect;
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;

pub use root::Root;
pub use window_components::*;

use crate::app::App;

/// Core trait for all UI components
/// 
/// Every renderable component must implement this trait to participate
/// in the rendering and layout system.
pub trait Renderable {
    /// Z-order for render layering (lower = behind). Default: 10 (content).
    /// Order: content 10, sidebar 20, header 100.
    fn z_order(&self) -> i32 {
        10
    }

    /// Render this component to vertices.
    /// When dirty_rect is Some(r), only components that intersect r need to render; Root culls by bounds.
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>);
    
    /// Get the bounding rectangle of this component
    fn bounds(&self) -> Rect;
    
    /// Bounds when this component's rect is derived from app state (e.g. window position/size).
    /// Window components return Some(rect); others return None and Root uses bounds().
    fn bounds_from_app(&self, _app: &App) -> Option<Rect> {
        None
    }
    
    /// Handle hit testing - returns true if point is within this component
    /// Override for custom hit detection logic
    fn contains(&self, point: Vec2) -> bool {
        self.bounds().contains_point(point)
    }
    
    /// Update layout based on available space.
    /// When dirty_rect is Some(r), only components that intersect r need layout; Root culls by bounds.
    fn update_layout(&mut self, available_rect: Rect, dirty_rect: Option<Rect>, app: Option<&App>);
    
    /// Get minimum size this component needs
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
    
    /// Compute wrapped size with optional constraints
    /// This is used by containers to compute content-wrapped dimensions.
    /// Default implementation returns min_size().
    /// 
    /// `max_width` is the maximum width available (None = no constraint)
    /// `measure_fn` is an optional function to measure text: `fn(&str, f32) -> Vec2`
    fn wrapped_size(&self, max_width: Option<f32>, _measure_fn: Option<&mut dyn FnMut(&str, f32) -> Vec2>) -> Vec2 {
        // Default: just return min_size, constrained by max_width if provided
        let size = self.min_size();
        if let Some(max_w) = max_width {
            Vec2::new(size.x.min(max_w), size.y)
        } else {
            size
        }
    }
}

/// A container that stacks children vertically
pub struct VStack {
    pub children: Vec<Box<dyn Renderable>>,
    pub rect: Rect,
    pub spacing: f32,
    pub padding: f32,
}

impl VStack {
    pub fn new(spacing: f32, padding: f32) -> Self {
        Self {
            children: Vec::new(),
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            spacing,
            padding,
        }
    }
    
    pub fn add_child(&mut self, child: Box<dyn Renderable>) {
        self.children.push(child);
    }
    
    /// Compute the size needed to wrap all content
    /// This calculates the dimensions based on children's wrapped_size() with optional max width constraint
    /// 
    /// `max_width` is the maximum width for the container (None = no constraint)
    /// `measure_fn` is an optional function to measure text for word wrapping: `fn(&str, f32) -> Vec2`
    /// 
    /// Example for chat bubbles:
    /// ```rust,ignore
    /// let bubble_size = vstack.wrap_content(
    ///     Some(max_bubble_width),
    ///     Some(&mut |text, font_size| renderer.measure_text(text, font_size))
    /// );
    /// ```
    pub fn wrap_content(&self, max_width: Option<f32>, mut measure_fn: Option<&mut dyn FnMut(&str, f32) -> Vec2>) -> Vec2 {
        let mut total_height = self.padding * 2.0;
        let mut max_width_used: f32 = 0.0;
        
        // Calculate available width for children (accounting for padding)
        let available_width = max_width.map(|w| w - self.padding * 2.0);
        
        for (i, child) in self.children.iter().enumerate() {
            // Use wrapped_size which handles text wrapping if measure_fn is provided
            // We need to re-borrow measure_fn for each iteration
            // Since we can't move it, we create a new Option with a re-borrowed reference
            // Re-borrow measure_fn for each iteration
            let child_size = if let Some(mf) = measure_fn.as_mut() {
                child.wrapped_size(available_width, Some(*mf))
            } else {
                child.wrapped_size(available_width, None)
            };
            
            // Constrain child width if max_width is specified
            let child_width = if let Some(max_w) = available_width {
                child_size.x.min(max_w)
            } else {
                child_size.x
            };
            
            max_width_used = max_width_used.max(child_width);
            total_height += child_size.y;
            
            if i < self.children.len() - 1 {
                total_height += self.spacing;
            }
        }
        
        Vec2::new(
            max_width_used + self.padding * 2.0,
            total_height
        )
    }
    
    /// Add a Text component to this VStack
    /// This is the only way to create Text components - they must be part of a container
    pub fn add_text(&mut self, text: impl Into<String>) -> &mut Self {
        use crate::ui::text::{Text, TextAlignment, TextCreationToken};
        use crate::ui::style;
        let token = TextCreationToken::new();
        let text_component = Text::new_internal(text, token)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Left);
        self.children.push(Box::new(text_component));
        self
    }
    
    /// Add a Text component with custom styling
    pub fn add_text_styled(
        &mut self,
        text: impl Into<String>,
        font_size: f32,
        color: glam::Vec4,
        alignment: crate::ui::text::TextAlignment,
    ) -> &mut Self {
        use crate::ui::text::{Text, TextCreationToken};
        let token = TextCreationToken::new();
        let text_component = Text::new_internal(text, token)
            .with_font_size(font_size)
            .with_color(color)
            .with_alignment(alignment);
        self.children.push(Box::new(text_component));
        self
    }
    
    pub fn with_children(mut self, children: Vec<Box<dyn Renderable>>) -> Self {
        self.children = children;
        self
    }
}

impl Renderable for VStack {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        let component_id = format!("vstack_{:p}", self);
        renderer.validate_component(&component_id, None, "VStack");
        renderer.push_parent(component_id.clone());
        for child in &self.children {
            child.render(renderer, app, vertices, dirty_rect);
        }
        renderer.pop_parent();
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, dirty_rect: Option<Rect>, app: Option<&App>) {
        self.rect = available_rect;
        let content_rect = available_rect.inset(self.padding);
        let mut current_y = content_rect.y;
        for child in &mut self.children {
            let child_min = child.min_size();
            let max_y = content_rect.bottom();
            if current_y >= max_y {
                break;
            }
            let child_rect = Rect::new(
                content_rect.x,
                current_y,
                content_rect.width,
                child_min.y.min(max_y - current_y),
            );
            child.update_layout(child_rect, dirty_rect, app);
            current_y += child_min.y + self.spacing;
            if current_y > max_y {
                break;
            }
        }
    }
    
    fn min_size(&self) -> Vec2 {
        let mut total_height = self.padding * 2.0;
        let mut max_width = 0.0;
        
        for (i, child) in self.children.iter().enumerate() {
            let child_size = child.min_size();
            total_height += child_size.y;
            max_width = f32::max(max_width, child_size.x);
            
            if i < self.children.len() - 1 {
                total_height += self.spacing;
            }
        }
        
        Vec2::new(max_width + self.padding * 2.0, total_height)
    }
}

/// A container that stacks children horizontally
pub struct HStack {
    pub children: Vec<Box<dyn Renderable>>,
    pub rect: Rect,
    pub spacing: f32,
    pub padding: f32,
}

impl HStack {
    pub fn new(spacing: f32, padding: f32) -> Self {
        Self {
            children: Vec::new(),
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            spacing,
            padding,
        }
    }
    
    pub fn add_child(&mut self, child: Box<dyn Renderable>) {
        self.children.push(child);
    }
    
    /// Add a Text component to this HStack
    /// This is the only way to create Text components - they must be part of a container
    pub fn add_text(&mut self, text: impl Into<String>) -> &mut Self {
        use crate::ui::text::{Text, TextAlignment, TextCreationToken};
        use crate::ui::style;
        let token = TextCreationToken::new();
        let text_component = Text::new_internal(text, token)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Left);
        self.children.push(Box::new(text_component));
        self
    }
    
    /// Add a Text component with custom styling
    pub fn add_text_styled(
        &mut self,
        text: impl Into<String>,
        font_size: f32,
        color: glam::Vec4,
        alignment: crate::ui::text::TextAlignment,
    ) -> &mut Self {
        use crate::ui::text::{Text, TextCreationToken};
        let token = TextCreationToken::new();
        let text_component = Text::new_internal(text, token)
            .with_font_size(font_size)
            .with_color(color)
            .with_alignment(alignment);
        self.children.push(Box::new(text_component));
        self
    }
    
    pub fn with_children(mut self, children: Vec<Box<dyn Renderable>>) -> Self {
        self.children = children;
        self
    }
    
    /// Compute the size needed to wrap all content
    /// This calculates the dimensions based on children's wrapped_size() with optional max height constraint
    /// 
    /// `max_height` is the maximum height for the container (None = no constraint)
    /// `measure_fn` is an optional function to measure text for word wrapping: `fn(&str, f32) -> Vec2`
    pub fn wrap_content(&self, max_height: Option<f32>, measure_fn: &mut Option<&mut dyn FnMut(&str, f32) -> Vec2>) -> Vec2 {
        let mut total_width = self.padding * 2.0;
        let mut max_height_used: f32 = 0.0;
        
        // Calculate available height for children (accounting for padding)
        let available_height = max_height.map(|h| h - self.padding * 2.0);
        
        for (i, child) in self.children.iter().enumerate() {
            // Use wrapped_size which handles text wrapping if measure_fn is provided
            // We need to re-borrow measure_fn for each iteration
            // Re-borrow measure_fn for each iteration
            let child_size = if let Some(mf) = measure_fn.as_mut() {
                child.wrapped_size(None, Some(*mf))
            } else {
                child.wrapped_size(None, None)
            }; // HStack doesn't constrain width
            
            // Constrain child height if max_height is specified
            let child_height = if let Some(max_h) = available_height {
                child_size.y.min(max_h)
            } else {
                child_size.y
            };
            
            max_height_used = max_height_used.max(child_height);
            total_width += child_size.x;
            
            if i < self.children.len() - 1 {
                total_width += self.spacing;
            }
        }
        
        Vec2::new(
            total_width,
            max_height_used + self.padding * 2.0
        )
    }
}

impl Renderable for HStack {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        let component_id = format!("hstack_{:p}", self);
        renderer.validate_component(&component_id, None, "HStack");
        renderer.push_parent(component_id.clone());
        for child in &self.children {
            child.render(renderer, app, vertices, dirty_rect);
        }
        renderer.pop_parent();
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, dirty_rect: Option<Rect>, app: Option<&App>) {
        self.rect = available_rect;
        let content_rect = available_rect.inset(self.padding);
        let mut current_x = content_rect.x;
        for child in &mut self.children {
            let child_min = child.min_size();
            let child_rect = Rect::new(
                current_x,
                content_rect.y,
                child_min.x,
                content_rect.height,
            );
            child.update_layout(child_rect, dirty_rect, app);
            current_x += child_min.x + self.spacing;
        }
    }
    
    fn min_size(&self) -> Vec2 {
        let mut total_width = self.padding * 2.0;
        let mut max_height = 0.0;
        
        for (i, child) in self.children.iter().enumerate() {
            let child_size = child.min_size();
            total_width += child_size.x;
            max_height = f32::max(max_height, child_size.y);
            
            if i < self.children.len() - 1 {
                total_width += self.spacing;
            }
        }
        
        Vec2::new(total_width, max_height + self.padding * 2.0)
    }
}

/// Builder for creating UI hierarchies
pub struct UIBuilder;

impl UIBuilder {
    /// Create a vertical stack
    pub fn vstack(spacing: f32, padding: f32) -> VStack {
        VStack::new(spacing, padding)
    }
    
    /// Create a horizontal stack
    pub fn hstack(spacing: f32, padding: f32) -> HStack {
        HStack::new(spacing, padding)
    }
}
