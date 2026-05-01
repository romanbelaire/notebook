use glam::Vec2;
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::shadow::ShadowSpec;
use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ButtonState {
    Normal,
    Hover,
    Pressed,
}

#[derive(Clone)]
pub struct Button {
    pub position: Vec2,
    pub size: Vec2,
    pub label: String,
    pub state: ButtonState,
    pub shadow: Option<ShadowSpec>,
}

impl Button {
    pub fn new(position: Vec2, size: Vec2, label: &str) -> Self {
        Self {
            position,
            size,
            label: label.to_string(),
            state: ButtonState::Normal,
            shadow: None,
        }
    }

    /// Attach a drop shadow. See [`crate::ui::style::elevation`] for presets.
    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
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

impl Renderable for Button {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, _dirty_rect: Option<Rect>) {
        use crate::ui::style;
        use crate::ui::icons::icon_names;
        use glam::Vec4;
        use crate::ui::{Text, TextAlignment};

        let component_id = format!("button_{:p}", self as *const Self);
        renderer.validate_component(&component_id, None, "Button");

        let button_rect = Rect::from_pos_size(self.position, self.size);

        if let Some(spec) = &self.shadow {
            renderer.queue_shadow(&button_rect, style::corner_radius::MEDIUM, spec);
        }

        let bg_color = match self.state {
            ButtonState::Pressed => style::button::PRIMARY() * Vec4::new(1.0, 1.0, 1.0, 0.8),
            ButtonState::Hover => style::button::PRIMARY() * Vec4::new(1.0, 1.0, 1.0, 0.9),
            ButtonState::Normal => style::button::PRIMARY(),
        };

        let button_bg = Quad {
            position: self.position,
            size: self.size,
            color: bg_color,
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&button_bg.to_vertices());

        let icon_name = match self.label.as_str() {
            "__plus" | "Add" | "+" => Some(icon_names::PLUS),
            "__save" => Some(icon_names::SAVE),
            "__open" => Some(icon_names::FOLDER),
            "__delete" | "Delete" | "Trash" => Some(icon_names::TRASH),
            "Close" | "✕" => Some(icon_names::CLOSE),
            "Edit" => Some(icon_names::PENCIL),
            "Search" => Some(icon_names::MAGNIFY),
            _ => None,
        };

        if let Some(icon) = icon_name {
            let icon_size = 16.0;
            let icon_pos = Vec2::new(
                button_rect.x + button_rect.width / 2.0 - icon_size / 2.0,
                button_rect.y + button_rect.height / 2.0 - icon_size / 2.0,
            );
            renderer.queue_icon(icon, icon_pos, icon_size, style::text::PRIMARY());
        } else {
            let text_rect = Rect::from_pos_size(button_rect.position(), button_rect.size());
            let mut button_text = Text::new_for_render(&self.label)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY())
                .with_alignment(TextAlignment::Center);
            button_text.update_layout(text_rect, None, None);
            renderer.push_parent(component_id.clone());
            button_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    }

    fn bounds(&self) -> Rect {
        Rect::new(self.position.x, self.position.y, self.size.x, self.size.y)
    }

    fn update_layout(&mut self, available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        self.position = available_rect.position();
        self.size = available_rect.size();
    }

    fn min_size(&self) -> Vec2 {
        Vec2::new(self.size.x.max(1.0), self.size.y.max(1.0))
    }

    fn corner_radius(&self) -> f32 {
        crate::ui::style::corner_radius::MEDIUM
    }
}

