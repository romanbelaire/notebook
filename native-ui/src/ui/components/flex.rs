//! Flex children for [super::HStack] and [super::VStack].
//!
//! Wrap a child in [`Expanded`] so the stack shares remaining space among flex children
//! (proportional to `weight`). Children without [`Renderable::flex_weight`] use fixed
//! [`Renderable::min_size`] along the stack axis.

use glam::Vec2;
use crate::ui::core::Rect;
use crate::ui::shadow::ShadowSpec;
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use crate::app::App;
use super::Renderable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlexAxis {
    /// Flex along width ([super::HStack]).
    Horizontal,
    /// Flex along height ([super::VStack]).
    Vertical,
}

/// Takes remaining space in the parent stack along [`FlexAxis`].
pub struct Expanded {
    pub axis: FlexAxis,
    pub weight: f32,
    pub child: Box<dyn Renderable>,
    pub shadow: Option<ShadowSpec>,
}

impl Expanded {
    pub fn horizontal(weight: f32, child: Box<dyn Renderable>) -> Self {
        Self {
            axis: FlexAxis::Horizontal,
            weight,
            child,
            shadow: None,
        }
    }

    pub fn vertical(weight: f32, child: Box<dyn Renderable>) -> Self {
        Self {
            axis: FlexAxis::Vertical,
            weight,
            child,
            shadow: None,
        }
    }

    /// Attach a drop shadow behind the child's bounds.
    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for Expanded {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        if let Some(spec) = &self.shadow {
            renderer.queue_shadow(&self.child.bounds(), self.child.corner_radius(), spec);
        }
        self.child.render(renderer, app, vertices, dirty_rect);
    }

    fn bounds(&self) -> Rect {
        self.child.bounds()
    }

    fn update_layout(&mut self, available_rect: Rect, dirty_rect: Option<Rect>, app: Option<&App>) {
        self.child.update_layout(available_rect, dirty_rect, app);
    }

    fn min_size(&self) -> Vec2 {
        let inner = self.child.min_size();
        match self.axis {
            FlexAxis::Horizontal => Vec2::new(0.0, inner.y),
            FlexAxis::Vertical => Vec2::new(inner.x, 0.0),
        }
    }

    fn flex_weight(&self) -> Option<f32> {
        Some(self.weight)
    }

    fn corner_radius(&self) -> f32 {
        self.child.corner_radius()
    }
}
