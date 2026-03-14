# Sidebar Container System Example

## New Approach with SectionStack

The sidebar now uses a proper container/section system that handles:
- Section titles (Conversations, Documents, Insights)
- Scrollable content within each section
- Proper spacing and layout
- Hit testing for clicks

## Usage Example

```rust
use crate::ui::core::container::{Section, SectionStack};

// Create section stack
let mut stack = SectionStack::new(spacing: 16.0);

// Add Conversations section
let mut conversations_section = Section::new("Conversations".to_string(), item_height: 40.0);
conversations_section.item_count = app.chat_state.conversations.len();
conversations_section.max_content_height = Some(300.0);  // Max 300px tall
conversations_section.scroll_offset = app.sidebar.conversations_scroll;
stack.add_section(conversations_section);

// Add Documents section
let mut documents_section = Section::new("Documents".to_string(), item_height: 40.0);
documents_section.item_count = document_ids.len();
documents_section.max_content_height = Some(250.0);
documents_section.scroll_offset = app.sidebar.documents_scroll;
stack.add_section(documents_section);

// Add Insights section (non-scrollable, shows all)
let mut insights_section = Section::new("Insights".to_string(), item_height: 60.0);
insights_section.item_count = app.insights_state.insights.len();
insights_section.scrollable = false;
insights_section.max_content_height = None;  // Show all
stack.add_section(insights_section);

// Layout sections
let sidebar_rect = Rect::new(x, y, width, height);
let layout = stack.layout(&sidebar_rect);

// Render each section
for (section_idx, y_offset) in layout {
    let section = &stack.sections[section_idx];
    
    // Render title
    let title_rect = section.title_rect(&sidebar_rect, y_offset);
    render_title(&title_rect, &section.title);
    
    // Render items
    for item_idx in 0..section.item_count {
        if let Some(item_rect) = section.item_rect(&sidebar_rect, y_offset, item_idx, padding) {
            render_item(&item_rect, item_idx);
        }
    }
}

// Hit testing
if let Some(hit) = stack.hit_test(&sidebar_rect, mouse_pos, padding) {
    match hit {
        SectionHit::Title(section_idx) => {
            // Clicked on section title
        }
        SectionHit::Item(section_idx, item_idx) => {
            // Clicked on item
            if section_idx == 0 {
                // Conversations
                select_conversation(item_idx);
            } else if section_idx == 1 {
                // Documents
                select_document(item_idx);
            }
        }
        SectionHit::Content(section_idx) => {
            // Clicked in scrollable area but not on item
        }
    }
}
```

## Benefits

1. **Automatic Layout**: No more manual Y calculations
2. **Scrolling**: Each section can scroll independently
3. **Hit Testing**: Built-in click detection
4. **Flexible**: Easy to add/remove/reorder sections
5. **Spacing**: Consistent spacing between sections

## Integration with Existing Code

The sidebar.rs structure stays mostly the same, but rendering uses the new system:

```rust
// In sidebar.rs
pub struct SidebarWindow {
    // ... existing fields ...
    
    // Add section stack
    pub section_stack: SectionStack,
    
    // Scroll offsets for each section
    pub conversations_scroll: f32,
    pub documents_scroll: f32,
}

impl SidebarWindow {
    pub fn rebuild_sections(&mut self, app: &App) {
        self.section_stack = SectionStack::new(16.0);
        
        // Build conversations section
        let mut conv_section = Section::new("Conversations".to_string(), 40.0);
        conv_section.item_count = app.chat_state.conversations.len();
        conv_section.max_content_height = Some(300.0);
        conv_section.scroll_offset = self.conversations_scroll;
        self.section_stack.add_section(conv_section);
        
        // ... build other sections ...
    }
}
```

## Rendering Pattern

```rust
pub fn render_sidebar_content(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if !app.sidebar.is_open {
        return;
    }
    
    let sidebar_rect = Rect::new(x, y, width, height);
    let padding = 12.0;
    
    // Get layout
    let layout = app.sidebar.section_stack.layout(&sidebar_rect);
    
    for (section_idx, y_offset) in layout {
        let section = &app.sidebar.section_stack.sections[section_idx];
        
        // Render section title
        let title_rect = section.title_rect(&sidebar_rect, y_offset);
        let title_pos = text::left_aligned(&title_rect, 14.0, padding);
        renderer.queue_text(&section.title, title_pos, style::text::PRIMARY, 14.0);
        
        // Render items
        match section_idx {
            0 => render_conversations(app, &sidebar_rect, y_offset, section, vertices, renderer),
            1 => render_documents(app, &sidebar_rect, y_offset, section, vertices, renderer),
            2 => render_insights(app, &sidebar_rect, y_offset, section, vertices, renderer),
            _ => {}
        }
    }
}
```

This provides a clean, modular system for the sidebar!

