use glam::Vec2;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Toast {
    pub id: u64,
    pub message: String,
    pub toast_type: ToastType,
    pub created_at: Instant,
    pub position: Vec2,
    pub opacity: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToastType {
    Info,
    Success,
    Error,
}

pub struct ToastManager {
    pub toasts: Vec<Toast>,
    next_id: u64,
}

impl ToastManager {
    pub fn new() -> Self {
        Self {
            toasts: Vec::new(),
            next_id: 0,
        }
    }

    pub fn show(&mut self, message: String, toast_type: ToastType, viewport_size: Vec2) {
        let id = self.next_id;
        self.next_id += 1;
        
        let toast = Toast {
            id,
            message,
            toast_type,
            created_at: Instant::now(),
            position: Vec2::new(viewport_size.x - 250.0, viewport_size.y - 100.0 - (self.toasts.len() as f32 * 60.0)),
            opacity: 0.0, // Start invisible, fade in
        };
        
        self.toasts.push(toast);
    }

    pub fn update(&mut self, dt: f32, viewport_size: Vec2) {
        let now = Instant::now();
        let toast_lifetime = 4.0; // 4 seconds
        let fade_in_duration = 0.2; // 0.2 seconds to fade in
        let fade_out_duration = 0.3; // 0.3 seconds to fade out
        
        // Update positions and opacity
        let toast_count = self.toasts.len();
        for (i, toast) in self.toasts.iter_mut().enumerate() {
            let elapsed = now.duration_since(toast.created_at).as_secs_f32();
            
            // Update position (stack from bottom)
            toast.position = Vec2::new(
                viewport_size.x - 250.0,
                viewport_size.y - 100.0 - ((toast_count - i - 1) as f32 * 60.0),
            );
            
            // Update opacity
            if elapsed < fade_in_duration {
                // Fade in
                toast.opacity = (elapsed / fade_in_duration).min(1.0);
            } else if elapsed > toast_lifetime - fade_out_duration {
                // Fade out
                let fade_out_elapsed = elapsed - (toast_lifetime - fade_out_duration);
                toast.opacity = (1.0 - (fade_out_elapsed / fade_out_duration)).max(0.0);
            } else {
                // Fully visible
                toast.opacity = 1.0;
            }
        }
        
        // Remove expired toasts
        self.toasts.retain(|toast| {
            let elapsed = now.duration_since(toast.created_at).as_secs_f32();
            elapsed < toast_lifetime
        });
    }

    pub fn remove(&mut self, id: u64) {
        self.toasts.retain(|t| t.id != id);
    }
}

impl Default for ToastManager {
    fn default() -> Self {
        Self::new()
    }
}

