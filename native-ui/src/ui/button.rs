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

