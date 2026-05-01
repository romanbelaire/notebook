use glam::Vec2;
use std::time::Instant;

use crate::ui::style;

#[derive(Clone, Debug)]
pub struct Toast {
    pub id: u64,
    pub message: String,
    pub toast_type: ToastType,
    pub created_at: Instant,
    pub position: Vec2,
    pub opacity: f32,
    /// Layout height for bottom stacking (from line heuristic).
    pub height: f32,
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

    fn estimate_toast_height(message: &str) -> f32 {
        let chars = message.chars().count() as f32;
        let lines = (chars / style::toast::HEURISTIC_CHARS_PER_LINE).ceil().max(1.0);
        let body_h = lines * (style::font_size::TOOLTIP * style::font_size::LINE_HEIGHT_RATIO);
        (style::toast::CARD_PADDING * 2.0 + body_h)
            .max(style::toast::CARD_MIN_HEIGHT)
            .min(style::toast::CARD_MAX_HEIGHT)
    }

    pub fn show(&mut self, message: String, toast_type: ToastType, viewport_size: Vec2) {
        let id = self.next_id;
        self.next_id += 1;

        let height = Self::estimate_toast_height(&message);
        let toast = Toast {
            id,
            message,
            toast_type,
            created_at: Instant::now(),
            position: Vec2::ZERO,
            opacity: 0.0,
            height,
        };

        self.toasts.push(toast);
        self.layout_toast_positions(viewport_size);
    }

    fn layout_toast_positions(&mut self, viewport_size: Vec2) {
        let mut acc = style::toast::MARGIN_Y;
        for i in (0..self.toasts.len()).rev() {
            let h = self.toasts[i].height;
            self.toasts[i].position = Vec2::new(
                viewport_size.x - style::toast::CARD_WIDTH - style::toast::MARGIN_X,
                viewport_size.y - acc - h,
            );
            acc += h + style::toast::STACK_GAP;
        }
    }

    pub fn update(&mut self, dt: f32, viewport_size: Vec2) {
        let now = Instant::now();
        let toast_lifetime = 4.0; // 4 seconds
        let fade_in_duration = 0.2; // 0.2 seconds to fade in
        let fade_out_duration = 0.3; // 0.3 seconds to fade out
        
        // Update positions and opacity
        for toast in self.toasts.iter_mut() {
            let elapsed = now.duration_since(toast.created_at).as_secs_f32();

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

        self.layout_toast_positions(viewport_size);
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

