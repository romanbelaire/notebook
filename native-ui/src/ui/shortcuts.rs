use winit::keyboard::{KeyCode, ModifiersState};

/// Keyboard shortcut identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShortcutId(pub usize);

/// Keyboard shortcut definition
#[derive(Debug, Clone)]
pub struct Shortcut {
    pub id: ShortcutId,
    pub modifiers: ModifiersState,
    pub key: KeyCode,
    pub description: String,
}

/// Centralized keyboard shortcut registry
pub struct ShortcutRegistry {
    shortcuts: Vec<Shortcut>,
    next_id: usize,
}

impl ShortcutRegistry {
    pub fn new() -> Self {
        Self {
            shortcuts: Vec::new(),
            next_id: 0,
        }
    }
    
    /// Register a new shortcut
    pub fn register(&mut self, modifiers: ModifiersState, key: KeyCode, description: String) -> ShortcutId {
        let id = ShortcutId(self.next_id);
        self.next_id += 1;
        self.shortcuts.push(Shortcut {
            id,
            modifiers,
            key,
            description,
        });
        id
    }
    
    /// Find shortcut matching key and modifiers
    pub fn find(&self, modifiers: ModifiersState, key: KeyCode) -> Option<ShortcutId> {
        self.shortcuts.iter()
            .find(|s| s.modifiers == modifiers && s.key == key)
            .map(|s| s.id)
    }
    
    /// Get all registered shortcuts
    pub fn all(&self) -> &[Shortcut] {
        &self.shortcuts
    }
    
    /// Check if a shortcut conflicts with existing ones
    pub fn conflicts(&self, modifiers: ModifiersState, key: KeyCode) -> bool {
        self.shortcuts.iter()
            .any(|s| s.modifiers == modifiers && s.key == key)
    }
}

impl Default for ShortcutRegistry {
    fn default() -> Self {
        Self::new()
    }
}

