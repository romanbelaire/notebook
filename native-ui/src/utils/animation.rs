/// Spring physics animation for smooth, natural motion
pub struct SpringAnimation {
    pub value: f32,
    pub target: f32,
    pub velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
}

/// Preset animation configurations for common use cases
#[derive(Debug, Clone, Copy)]
pub enum AnimationPreset {
    /// Smooth, gentle animation (default)
    Smooth,
    /// Fast, responsive animation
    Snappy,
    /// Bouncy animation with tight bounce
    Bouncy,
    /// Very fast, tight bounce (for cursor)
    TightBounce,
    /// Slow, gentle animation
    Gentle,
    /// Viscous: slow, high stretch, no bounce (for liquid nav slider trailing edge)
    Viscous,
    /// Custom parameters
    Custom { stiffness: f32, damping: f32 },
}

impl AnimationPreset {
    pub fn stiffness(&self) -> f32 {
        match self {
            AnimationPreset::Smooth => 200.0,
            AnimationPreset::Snappy => 400.0,
            AnimationPreset::Bouncy => 300.0,
            AnimationPreset::TightBounce => 800.0,
            AnimationPreset::Gentle => 100.0,
            AnimationPreset::Viscous => 65.0,
            AnimationPreset::Custom { stiffness, .. } => *stiffness,
        }
    }

    pub fn damping(&self) -> f32 {
        match self {
            AnimationPreset::Smooth => 15.0,
            AnimationPreset::Snappy => 20.0,
            AnimationPreset::Bouncy => 12.0,
            AnimationPreset::TightBounce => 8.0,
            AnimationPreset::Gentle => 10.0,
            AnimationPreset::Viscous => 22.0,
            AnimationPreset::Custom { damping, .. } => *damping,
        }
    }
}

impl Default for AnimationPreset {
    fn default() -> Self {
        AnimationPreset::Smooth
    }
}

impl SpringAnimation {
    /// Create a new spring animation with default smooth preset
    pub fn new(initial: f32) -> Self {
        Self::with_preset(initial, AnimationPreset::Smooth)
    }

    /// Create a new spring animation with a preset configuration
    pub fn with_preset(initial: f32, preset: AnimationPreset) -> Self {
        Self {
            value: initial,
            target: initial,
            velocity: 0.0,
            stiffness: preset.stiffness(),
            damping: preset.damping(),
        }
    }

    /// Create a new spring animation with custom parameters
    pub fn with_params(initial: f32, stiffness: f32, damping: f32) -> Self {
        Self {
            value: initial,
            target: initial,
            velocity: 0.0,
            stiffness,
            damping,
        }
    }

    /// Update the animation with delta time
    pub fn update(&mut self, dt: f32) {
        let force = (self.target - self.value) * self.stiffness;
        self.velocity += force * dt;
        self.velocity *= 1.0 - (self.damping * dt).min(0.9);
        self.value += self.velocity * dt;

        if (self.target - self.value).abs() < 0.01 && self.velocity.abs() < 0.01 {
            self.value = self.target;
            self.velocity = 0.0;
        }
    }

    /// Set the target value for the animation
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Check if the animation has reached its target
    pub fn is_at_target(&self) -> bool {
        (self.value - self.target).abs() < 0.01
    }

    /// Get the current interpolated value
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Get the target value
    pub fn target(&self) -> f32 {
        self.target
    }
}

/// Trait for values that can be animated
pub trait Animatable {
    type Value: Copy;
    
    /// Get the current animated value
    fn animated_value(&self) -> Self::Value;
    
    /// Get the target value
    fn target_value(&self) -> Self::Value;
    
    /// Set the target value
    fn set_target(&mut self, target: Self::Value);
    
    /// Update the animation with delta time
    fn update(&mut self, dt: f32);
    
    /// Check if the animation has reached its target
    fn is_at_target(&self) -> bool;
}

impl Animatable for SpringAnimation {
    type Value = f32;
    
    fn animated_value(&self) -> f32 {
        self.value
    }
    
    fn target_value(&self) -> f32 {
        self.target
    }
    
    fn set_target(&mut self, target: f32) {
        SpringAnimation::set_target(self, target);
    }
    
    fn update(&mut self, dt: f32) {
        SpringAnimation::update(self, dt);
    }
    
    fn is_at_target(&self) -> bool {
        SpringAnimation::is_at_target(self)
    }
}

