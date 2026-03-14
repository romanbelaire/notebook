use glam::Vec2;
use winit::event::MouseButton;

/// Trait for components that can handle hover events
pub trait Hoverable {
    /// Called when mouse enters the component
    fn on_mouse_enter(&mut self, position: Vec2);
    
    /// Called when mouse leaves the component
    fn on_mouse_leave(&mut self);
    
    /// Check if point is within this component's bounds
    fn contains(&self, pos: Vec2) -> bool;
}

/// Trait for components that can handle drag operations
pub trait Draggable {
    /// Called when drag starts
    fn on_drag_start(&mut self, position: Vec2, button: MouseButton);
    
    /// Called during drag
    fn on_drag(&mut self, position: Vec2);
    
    /// Called when drag ends
    fn on_drag_end(&mut self, position: Vec2);
}

/// Mouse button type for event handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButtonType {
    Left,
    Right,
    Middle,
    Other(u16),
}

impl From<MouseButton> for MouseButtonType {
    fn from(button: MouseButton) -> Self {
        match button {
            MouseButton::Left => MouseButtonType::Left,
            MouseButton::Right => MouseButtonType::Right,
            MouseButton::Middle => MouseButtonType::Middle,
            MouseButton::Back => MouseButtonType::Other(0),
            MouseButton::Forward => MouseButtonType::Other(1),
            MouseButton::Other(code) => MouseButtonType::Other(code),
        }
    }
}

/// Drag operation state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DragState {
    None,
    Starting { button: MouseButtonType, start_pos: Vec2 },
    Dragging { button: MouseButtonType, start_pos: Vec2 },
}

/// Hover tracking state
#[derive(Debug, Clone)]
pub struct HoverState {
    pub hovered_component_id: Option<String>,
    pub last_hovered_component_id: Option<String>,
}

impl Default for HoverState {
    fn default() -> Self {
        Self {
            hovered_component_id: None,
            last_hovered_component_id: None,
        }
    }
}

/// Trait for components that can receive focus
pub trait Focusable {
    /// Focus this component
    fn focus(&mut self);
    
    /// Blur this component
    fn blur(&mut self);
    
    /// Check if this component is focused
    fn is_focused(&self) -> bool;
    
    /// Get component ID for focus traversal
    fn focus_id(&self) -> String;
}

/// Focus traversal direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocusDirection {
    Forward,  // Tab
    Backward, // Shift+Tab
}

/// Focus management state
#[derive(Debug, Clone)]
pub struct FocusState {
    pub focused_component_id: Option<String>,
    pub focusable_components: Vec<String>,  // Ordered list of focusable components
    pub focus_index: Option<usize>,  // Current index in focusable_components
}

impl Default for FocusState {
    fn default() -> Self {
        Self {
            focused_component_id: None,
            focusable_components: Vec::new(),
            focus_index: None,
        }
    }
}

/// Accessibility action types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessibilityAction {
    Activate,
    Increment,
    Decrement,
    ShowMenu,
    HideMenu,
    ScrollUp,
    ScrollDown,
    ScrollLeft,
    ScrollRight,
}

/// Accessibility state for components
#[derive(Debug, Clone)]
pub struct AccessibilityState {
    pub label: Option<String>,
    pub role: Option<String>,  // ARIA role
    pub value: Option<String>,  // ARIA value
    pub description: Option<String>,  // ARIA description
    pub accessible: bool,  // Whether component is accessible
}

impl Default for AccessibilityState {
    fn default() -> Self {
        Self {
            label: None,
            role: None,
            value: None,
            description: None,
            accessible: true,
        }
    }
}

/// Trait for accessible components
pub trait Accessible {
    /// Get accessibility state
    fn accessibility_state(&self) -> AccessibilityState;
    
    /// Handle accessibility action
    fn on_accessibility_action(&mut self, action: AccessibilityAction);
    
    /// Get accessibility label for screen readers
    fn accessibility_label(&self) -> Option<String> {
        self.accessibility_state().label
    }
}

