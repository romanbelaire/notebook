use glam::Vec2;

pub struct SubWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub velocity: Vec2,
    pub title: String,
    pub z_order: u32,
}

impl SubWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        Self {
            position,
            size,
            velocity: Vec2::ZERO,
            title: String::new(),
            z_order: 0,
        }
    }

    pub fn set_title(&mut self, title: String) {
        self.title = title;
    }

    pub fn hit_title_bar(&self, p: Vec2) -> bool {
        let rel = p - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= 30.0
    }

    pub fn update(&mut self, dt: f32, viewport: Vec2) {
        self.position += self.velocity * dt;

        if self.position.x < 0.0 {
            self.position.x = 0.0;
            self.velocity.x = 0.0;
        }
        if self.position.y < 0.0 {
            self.position.y = 0.0;
            self.velocity.y = 0.0;
        }
        if self.position.x + self.size.x > viewport.x {
            self.position.x = viewport.x - self.size.x;
            self.velocity.x = 0.0;
        }
        if self.position.y + self.size.y > viewport.y {
            self.position.y = viewport.y - self.size.y;
            self.velocity.y = 0.0;
        }
    }
}

