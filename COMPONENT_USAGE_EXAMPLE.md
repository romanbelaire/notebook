# Component Usage Examples

This document demonstrates how to use the new component-based architecture to build UI elements in a modular, composable way.

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Building a Sidebar](#building-a-sidebar)
3. [Custom Components](#custom-components)
4. [State Management](#state-management)
5. [Migration Guide](#migration-guide)

---

## Basic Concepts

### The Renderable Trait

Every UI component implements the `Renderable` trait:

```rust
pub trait Renderable {
    fn render(&self, renderer: &mut Renderer, vertices: &mut Vec<Vertex>);
    fn bounds(&self) -> Rect;
    fn update_layout(&mut self, available_rect: Rect);
    fn min_size(&self) -> Vec2;
    fn contains(&self, point: Vec2) -> bool { /* default impl */ }
}
```

### Container Components

- **VStack**: Stacks children vertically
- **HStack**: Stacks children horizontally
- **Section**: Title bar with content
- **ScrollableList**: Scrollable list of items

---

## Building a Sidebar

### Old Approach (Hardcoded)

```rust
// ❌ BAD: Hardcoded structure
pub struct SidebarWindow {
    pub conversations_list: ScrollView,
    pub documents_list: ScrollView,
    pub insights_panel: InsightsPanel,
    pub new_conversation_button: Button,
    pub new_document_button: Button,
    // ... many more hardcoded fields
}

impl SidebarWindow {
    pub fn new(position: Vec2, height: f32) -> Self {
        // Hardcoded Y calculations
        let list_height = height / 3.0;
        let conversations_y = 60.0;
        let documents_y = conversations_y + list_height;
        let insights_y = documents_y + list_height;
        
        // Must manually position everything
        Self {
            conversations_list: ScrollView::new(
                Vec2::new(position.x + 10.0, conversations_y),
                Vec2::new(280.0, list_height),
            ),
            documents_list: ScrollView::new(
                Vec2::new(position.x + 10.0, documents_y),
                Vec2::new(280.0, list_height),
            ),
            // ... more manual positioning
        }
    }
}
```

**Problems:**
- Can't add new sections without modifying struct
- Manual Y coordinate calculations
- Layout doesn't adapt to content
- Hard to reorder sections
- Tightly coupled

### New Approach (Component-Based)

```rust
// ✅ GOOD: Component-based structure
use crate::ui::components::{VStack, Renderable};
use crate::ui::components::sidebar::{Section, ScrollableList, SidebarBuilder};

pub struct SidebarWindow {
    pub root: VStack,
    pub position: Vec2,
    pub width: f32,
    pub height: f32,
    pub is_open: bool,
}

impl SidebarWindow {
    pub fn new(position: Vec2, height: f32, app_state: &AppState) -> Self {
        // Build sidebar declaratively from data
        let root = SidebarBuilder::new()
            .add_section(Box::new(
                Section::new(
                    "Conversations",
                    Box::new(ScrollableList::new(
                        app_state.conversations.clone(),
                        40.0, // item height
                        render_conversation_item,
                    ))
                )
            ))
            .add_section(Box::new(
                Section::new(
                    "Documents",
                    Box::new(ScrollableList::new(
                        app_state.documents.clone(),
                        40.0,
                        render_document_item,
                    ))
                )
            ))
            .add_section(Box::new(
                Section::new(
                    "Insights",
                    Box::new(ScrollableList::new(
                        app_state.insights.clone(),
                        35.0,
                        render_insight_item,
                    ))
                )
            ))
            .build();
        
        let mut sidebar = Self {
            root,
            position,
            width: 288.0,
            height,
            is_open: true,
        };
        
        // Layout automatically handles positioning
        sidebar.update_layout();
        sidebar
    }
    
    pub fn update_layout(&mut self) {
        let available = Rect::new(
            self.position.x,
            self.position.y,
            self.width,
            self.height,
        );
        self.root.update_layout(available);
    }
    
    pub fn render(&self, renderer: &mut Renderer, vertices: &mut Vec<Vertex>) {
        self.root.render(renderer, vertices);
    }
}
```

**Benefits:**
- Add sections by adding to array
- Automatic layout
- Reorder by changing array order
- Content-driven sizing
- Loosely coupled
- Easy to test

### Even Better: Fully Declarative

```rust
impl SidebarWindow {
    pub fn from_sections(
        position: Vec2,
        height: f32,
        sections: Vec<SectionConfig>,
    ) -> Self {
        let mut builder = SidebarBuilder::new();
        
        for config in sections {
            builder = builder.add_section(config.build());
        }
        
        let mut sidebar = Self {
            root: builder.build(),
            position,
            width: 288.0,
            height,
            is_open: true,
        };
        
        sidebar.update_layout();
        sidebar
    }
}

// Usage: initialize from configuration
let sidebar = SidebarWindow::from_sections(
    Vec2::new(0.0, 60.0),
    600.0,
    vec![
        SectionConfig {
            title: "Conversations",
            data: SectionData::Conversations(app.conversations.clone()),
            max_height: Some(200.0),
        },
        SectionConfig {
            title: "Documents",
            data: SectionData::Documents(app.documents.clone()),
            max_height: Some(200.0),
        },
        SectionConfig {
            title: "Insights",
            data: SectionData::Insights(app.insights.clone()),
            max_height: None,
        },
    ],
);
```

---

## Custom Components

### Creating a Custom Component

```rust
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use glam::Vec2;

pub struct CustomPanel {
    rect: Rect,
    title: String,
    content: String,
}

impl CustomPanel {
    pub fn new(title: String, content: String) -> Self {
        Self {
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            title,
            content,
        }
    }
}

impl Renderable for CustomPanel {
    fn render(&self, renderer: &mut Renderer, vertices: &mut Vec<Vertex>) {
        // Render background
        let bg = Quad {
            position: self.rect.position(),
            size: self.rect.size(),
            color: Vec4::new(0.2, 0.2, 0.2, 1.0),
            corner_radius: 8.0,
        };
        vertices.extend_from_slice(&bg.to_vertices());
        
        // Render title
        let title_pos = Vec2::new(
            self.rect.x + 10.0,
            self.rect.y + 10.0,
        );
        renderer.queue_text(&self.title, title_pos, Vec4::ONE, 16.0);
        
        // Render content
        let content_pos = Vec2::new(
            self.rect.x + 10.0,
            self.rect.y + 40.0,
        );
        renderer.queue_text(&self.content, content_pos, Vec4::ONE, 14.0);
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect) {
        self.rect = available_rect;
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(200.0, 100.0)
    }
}

// Use it anywhere
let panel = Box::new(CustomPanel::new(
    "Settings".to_string(),
    "Configure your preferences".to_string(),
));

// Add to a container
let mut vstack = VStack::new(10.0, 5.0);
vstack.add_child(panel);
```

### Composing Components

```rust
// Create a complex UI by composing simple components
let settings_panel = VStack::new(10.0, 10.0)
    .with_children(vec![
        Box::new(TitleBar::new("Settings")),
        Box::new(HStack::new(5.0, 5.0)
            .with_children(vec![
                Box::new(Label::new("Theme:")),
                Box::new(Dropdown::new(vec!["Dark", "Light"])),
            ])
        ),
        Box::new(HStack::new(5.0, 5.0)
            .with_children(vec![
                Box::new(Label::new("Font Size:")),
                Box::new(Slider::new(12.0, 24.0)),
            ])
        ),
        Box::new(HStack::new(5.0, 5.0)
            .with_children(vec![
                Box::new(Button::new("Save")),
                Box::new(Button::new("Cancel")),
            ])
        ),
    ]);
```

---

## State Management

### Separation of State and Rendering

```rust
// Define data structures separately from UI
pub struct ConversationData {
    pub id: String,
    pub title: String,
    pub last_updated: DateTime,
}

pub struct SidebarState {
    pub conversations: Vec<ConversationData>,
    pub selected_conversation: Option<String>,
    pub hovered_conversation: Option<usize>,
}

// UI components render based on state
impl SidebarWindow {
    pub fn update_from_state(&mut self, state: &SidebarState) {
        // Rebuild the list when data changes
        let conversations_list = ScrollableList::new(
            state.conversations.clone(),
            40.0,
            |conv, rect, renderer, vertices| {
                let is_selected = state.selected_conversation.as_ref() == Some(&conv.id);
                let is_hovered = state.hovered_conversation == Some(i);
                render_conversation_item(conv, rect, is_selected, is_hovered, renderer, vertices);
            },
        )
        .with_selected(state.selected_conversation.as_ref().and_then(|id| {
            state.conversations.iter().position(|c| &c.id == id)
        }))
        .with_hovered(state.hovered_conversation);
        
        // Update the section with new list
        // (In practice, you'd store references to update specific sections)
    }
}
```

### Event Handling

```rust
impl SidebarWindow {
    pub fn handle_click(&mut self, point: Vec2) -> Option<SidebarAction> {
        // Delegate to root component
        if !self.root.contains(point) {
            return None;
        }
        
        // Walk the component tree to find what was clicked
        // (This would be implemented in the Renderable trait)
        self.root.hit_test(point)
    }
}

pub enum SidebarAction {
    SelectConversation(String),
    SelectDocument(String),
    NewConversation,
    NewDocument,
}
```

---

## Migration Guide

### Step 1: Identify Hardcoded Structure

Look for:
- Manual Y coordinate calculations
- Hardcoded positioning in constructors
- Specific fields for each UI element
- Layout logic in constructors

### Step 2: Extract to Components

**Before:**
```rust
pub struct MyWindow {
    pub title_label: Label,
    pub item_list: ScrollView,
    pub action_button: Button,
}

impl MyWindow {
    pub fn new() -> Self {
        Self {
            title_label: Label::new(Vec2::new(10.0, 10.0), "Title"),
            item_list: ScrollView::new(Vec2::new(10.0, 40.0), Vec2::new(280.0, 400.0)),
            action_button: Button::new(Vec2::new(10.0, 450.0), Vec2::new(100.0, 30.0), "Action"),
        }
    }
}
```

**After:**
```rust
pub struct MyWindow {
    pub root: VStack,
}

impl MyWindow {
    pub fn new(data: &AppData) -> Self {
        let root = VStack::new(10.0, 10.0)
            .with_children(vec![
                Box::new(Label::new("Title")),
                Box::new(ScrollableList::new(data.items.clone(), 40.0, render_item)),
                Box::new(Button::new("Action")),
            ]);
        
        Self { root }
    }
}
```

### Step 3: Move Rendering to Components

**Before:**
```rust
pub fn render_my_window(window: &MyWindow, renderer: &mut Renderer, vertices: &mut Vec<Vertex>) {
    render_label(&window.title_label, renderer, vertices);
    render_scroll_view(&window.item_list, renderer, vertices);
    render_button(&window.action_button, renderer, vertices);
}
```

**After:**
```rust
impl MyWindow {
    pub fn render(&self, renderer: &mut Renderer, vertices: &mut Vec<Vertex>) {
        self.root.render(renderer, vertices);
    }
}
```

### Step 4: Update Layout Logic

**Before:**
```rust
pub fn update_layout(window: &mut MyWindow, height: f32) {
    window.title_label.position.y = 10.0;
    window.item_list.position.y = 40.0;
    window.item_list.size.y = height - 80.0;
    window.action_button.position.y = height - 40.0;
}
```

**After:**
```rust
impl MyWindow {
    pub fn update_layout(&mut self, available_rect: Rect) {
        self.root.update_layout(available_rect);
        // Layout is automatic!
    }
}
```

### Step 5: Simplify Hit Testing

**Before:**
```rust
pub fn handle_click(window: &MyWindow, point: Vec2) -> Option<Action> {
    if point.x >= window.action_button.position.x
        && point.x <= window.action_button.position.x + window.action_button.size.x
        && point.y >= window.action_button.position.y
        && point.y <= window.action_button.position.y + window.action_button.size.y
    {
        return Some(Action::ButtonClick);
    }
    
    if point.x >= window.item_list.position.x
        && point.x <= window.item_list.position.x + window.item_list.size.x
        && point.y >= window.item_list.position.y
        && point.y <= window.item_list.position.y + window.item_list.size.y
    {
        let index = ((point.y - window.item_list.position.y + window.item_list.scroll_offset) / 40.0) as usize;
        return Some(Action::SelectItem(index));
    }
    
    None
}
```

**After:**
```rust
impl MyWindow {
    pub fn handle_click(&self, point: Vec2) -> Option<Action> {
        // Components handle their own hit testing
        self.root.hit_test(point)
    }
}
```

---

## Best Practices

### ✅ DO

1. **Build from data**
   ```rust
   let sidebar = build_sidebar(&app_state);
   ```

2. **Use composition**
   ```rust
   VStack::new(10.0, 10.0)
       .with_children(vec![
           Box::new(Section::new(...)),
           Box::new(Section::new(...)),
       ])
   ```

3. **Keep components self-contained**
   ```rust
   impl Renderable for MyComponent {
       fn render(&self, ctx: &mut RenderContext) {
           // Only uses self, no external state
       }
   }
   ```

4. **Use type-safe IDs**
   ```rust
   pub struct ConversationId(String);
   pub struct DocumentId(String);
   ```

### ❌ DON'T

1. **Hardcode positions**
   ```rust
   // BAD
   let button = Button::new(Vec2::new(100.0, 200.0), size, "Click");
   ```

2. **Access parent/sibling state**
   ```rust
   // BAD
   impl Renderable for Child {
       fn render(&self, ctx: &mut RenderContext) {
           let parent_y = self.parent.position.y; // NO!
       }
   }
   ```

3. **Use manual layout calculations**
   ```rust
   // BAD
   let y1 = 10.0;
   let y2 = y1 + height1 + spacing;
   let y3 = y2 + height2 + spacing;
   ```

4. **Create god objects**
   ```rust
   // BAD
   struct Sidebar {
       conversations: Vec<Conversation>,
       documents: Vec<Document>,
       insights: Vec<Insight>,
       // ... 50 more fields
   }
   ```

---

## Summary

**Key Principles:**

1. **Modular**: Every UI element is a self-contained component
2. **Composable**: Build complex UIs from simple components
3. **Data-Driven**: Initialize from data, not hardcoded structure
4. **Encapsulated**: Components manage their own state and layout
5. **Flexible**: Easy to add, remove, or reorder elements

**The Result:**

- Less code
- Easier to maintain
- More flexible
- Better testability
- Clearer architecture

**When in doubt, ask:**

- Can I add a new section without modifying existing code? ✓
- Can I reorder sections by changing an array? ✓
- Can I test this component in isolation? ✓
- Is layout automatic? ✓
- Is positioning relative, not absolute? ✓

If you answer "yes" to all, you're doing it right! 🎉

