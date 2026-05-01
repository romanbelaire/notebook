use glam::Vec2;
use crate::ui::Button;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;
use crate::ui::shadow::ShadowSpec;
use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use crate::ui::style;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolbarButton {
    Bold,
    Italic,
    Underline,
    Strikethrough,
    Code,
    Link,
}

pub struct Toolbar {
    pub position: Vec2,
    pub size: Vec2,
    pub bold_button: Button,
    pub italic_button: Button,
    pub underline_button: Button,
    pub strikethrough_button: Button,
    pub code_button: Button,
    pub link_button: Button,
    pub button_size: Vec2,
    pub button_spacing: f32,
    pub shadow: Option<ShadowSpec>,
}

impl Toolbar {
    pub fn new(position: Vec2, width: f32) -> Self {
        let bx = style::toolbar_chrome::BUTTON_EXTENT;
        let button_size = Vec2::new(bx, bx);
        let toolbar_height = style::toolbar_chrome::BAR_HEIGHT;
        let button_spacing = style::toolbar_chrome::BUTTON_SPACING;
        
        Self {
            position,
            size: Vec2::new(width, toolbar_height),
            bold_button: Button::new(Vec2::ZERO, button_size, "B"),
            italic_button: Button::new(Vec2::ZERO, button_size, "I"),
            underline_button: Button::new(Vec2::ZERO, button_size, "U"),
            strikethrough_button: Button::new(Vec2::ZERO, button_size, "S"),
            code_button: Button::new(Vec2::ZERO, button_size, "</>"),
            link_button: Button::new(Vec2::ZERO, button_size, "🔗"),
            button_size,
            button_spacing,
            shadow: None,
        }
    }

    /// Attach a drop shadow behind the toolbar chassis.
    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }

    pub fn relayout_buttons(&mut self) {
        use crate::ui::core::layout;
        
        let toolbar_rect = Rect::new(
            self.position.x,
            self.position.y,
            self.size.x,
            self.size.y,
        );
        
        // Calculate button widths for horizontal stack
        let button_widths = [
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
        ];
        
        // Use stack_horizontal to position buttons, then vertically center in the bar
        let button_rects = layout::stack_horizontal(
            &toolbar_rect,
            &button_widths,
            self.button_spacing,
            0.0,
        );
        let dy = (toolbar_rect.height - self.button_size.y) * 0.5;
        
        if let Some(rect) = button_rects.get(0) {
            let p = rect.position();
            self.bold_button.position = Vec2::new(p.x, p.y + dy);
        }
        if let Some(rect) = button_rects.get(1) {
            let p = rect.position();
            self.italic_button.position = Vec2::new(p.x, p.y + dy);
        }
        if let Some(rect) = button_rects.get(2) {
            let p = rect.position();
            self.underline_button.position = Vec2::new(p.x, p.y + dy);
        }
        if let Some(rect) = button_rects.get(3) {
            let p = rect.position();
            self.strikethrough_button.position = Vec2::new(p.x, p.y + dy);
        }
        if let Some(rect) = button_rects.get(4) {
            let p = rect.position();
            self.code_button.position = Vec2::new(p.x, p.y + dy);
        }
        if let Some(rect) = button_rects.get(5) {
            let p = rect.position();
            self.link_button.position = Vec2::new(p.x, p.y + dy);
        }
    }

    pub fn hit_test(&self, pos: Vec2) -> Option<ToolbarButton> {
        if self.bold_button.contains(pos) {
            Some(ToolbarButton::Bold)
        } else if self.italic_button.contains(pos) {
            Some(ToolbarButton::Italic)
        } else if self.underline_button.contains(pos) {
            Some(ToolbarButton::Underline)
        } else if self.strikethrough_button.contains(pos) {
            Some(ToolbarButton::Strikethrough)
        } else if self.code_button.contains(pos) {
            Some(ToolbarButton::Code)
        } else if self.link_button.contains(pos) {
            Some(ToolbarButton::Link)
        } else {
            None
        }
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }

    pub fn on_hover(&mut self, pos: Vec2) {
        self.bold_button.on_hover(pos);
        self.italic_button.on_hover(pos);
        self.underline_button.on_hover(pos);
        self.strikethrough_button.on_hover(pos);
        self.code_button.on_hover(pos);
        self.link_button.on_hover(pos);
    }

    pub fn on_mouse_down(&mut self, pos: Vec2) {
        if self.bold_button.contains(pos) {
            self.bold_button.on_press();
        } else if self.italic_button.contains(pos) {
            self.italic_button.on_press();
        } else if self.underline_button.contains(pos) {
            self.underline_button.on_press();
        } else if self.strikethrough_button.contains(pos) {
            self.strikethrough_button.on_press();
        } else if self.code_button.contains(pos) {
            self.code_button.on_press();
        } else if self.link_button.contains(pos) {
            self.link_button.on_press();
        }
    }

    pub fn on_mouse_up(&mut self, pos: Vec2) {
        self.bold_button.on_release();
        self.italic_button.on_release();
        self.underline_button.on_release();
        self.strikethrough_button.on_release();
        self.code_button.on_release();
        self.link_button.on_release();
    }

    pub fn on_cancel(&mut self) {
        self.bold_button.on_cancel();
        self.italic_button.on_cancel();
        self.underline_button.on_cancel();
        self.strikethrough_button.on_cancel();
        self.code_button.on_cancel();
        self.link_button.on_cancel();
    }
}

impl Renderable for Toolbar {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        let component_id = format!("toolbar_{:p}", self as *const Self);
        renderer.validate_component(&component_id, None, "Toolbar");
        renderer.push_parent(component_id.clone());

        if let Some(spec) = &self.shadow {
            let bar_rect = Rect::new(self.position.x, self.position.y, self.size.x, self.size.y);
            renderer.queue_shadow(&bar_rect, style::corner_radius::SMALL, spec);
        }

        let toolbar_bg = Quad {
            position: self.position,
            size: self.size,
            color: style::bg::SECONDARY(),
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&toolbar_bg.to_vertices());

        Renderable::render(&self.bold_button, renderer, app, vertices, dirty_rect);
        Renderable::render(&self.italic_button, renderer, app, vertices, dirty_rect);
        Renderable::render(&self.underline_button, renderer, app, vertices, dirty_rect);
        Renderable::render(&self.strikethrough_button, renderer, app, vertices, dirty_rect);
        Renderable::render(&self.code_button, renderer, app, vertices, dirty_rect);
        Renderable::render(&self.link_button, renderer, app, vertices, dirty_rect);

        renderer.pop_parent();
    }

    fn bounds(&self) -> Rect {
        Rect::new(self.position.x, self.position.y, self.size.x, self.size.y)
    }

    fn update_layout(&mut self, available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        self.position = available_rect.position();
        self.size = available_rect.size();
        self.relayout_buttons();
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(
            6.0 * self.button_size.x + 5.0 * self.button_spacing,
            self.size.y.max(self.button_size.y),
        )
    }

    fn corner_radius(&self) -> f32 {
        style::corner_radius::SMALL
    }
}
