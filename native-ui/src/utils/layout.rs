use glam::Vec2;

pub struct LayoutHelper;

impl LayoutHelper {
    pub fn center_horizontal(parent_width: f32, child_width: f32) -> f32 {
        (parent_width - child_width) / 2.0
    }

    pub fn center_vertical(parent_height: f32, child_height: f32) -> f32 {
        (parent_height - child_height) / 2.0
    }

    pub fn align_right(parent_width: f32, child_width: f32) -> f32 {
        parent_width - child_width
    }

    pub fn align_bottom(parent_height: f32, child_height: f32) -> f32 {
        parent_height - child_height
    }
}

pub struct Stack {
    pub position: Vec2,
    pub spacing: f32,
    pub current_offset: f32,
}

impl Stack {
    pub fn new(position: Vec2, spacing: f32) -> Self {
        Self {
            position,
            spacing,
            current_offset: 0.0,
        }
    }

    pub fn add_item(&mut self, height: f32) -> Vec2 {
        let pos = Vec2::new(self.position.x, self.position.y + self.current_offset);
        self.current_offset += height + self.spacing;
        pos
    }

    pub fn reset(&mut self) {
        self.current_offset = 0.0;
    }
}

