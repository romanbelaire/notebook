/// Modular sidebar components using the Renderable architecture
/// 
/// This demonstrates how to build a sidebar from composable, reusable components
/// instead of hardcoding structure and layout.
use glam::{Vec2, Vec4};
use crate::ui::core::Rect;
use crate::ui::shadow::ShadowSpec;
use crate::ui::style;
use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use super::{Renderable, VStack};

/// A section with a title and scrollable content
pub struct Section {
    pub title: String,
    pub content: Box<dyn Renderable>,
    pub rect: Rect,
    pub title_height: f32,
    pub shadow: Option<ShadowSpec>,
}

impl Section {
    pub fn new(title: impl Into<String>, content: Box<dyn Renderable>) -> Self {
        Self {
            title: title.into(),
            content,
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            title_height: 40.0,
            shadow: None,
        }
    }

    /// Attach a drop shadow behind the section.
    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }
}

impl Renderable for Section {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        const TITLE_FONT_SIZE: f32 = style::font_size::NORMAL;
        if let Some(spec) = &self.shadow {
            renderer.queue_shadow(&self.rect, 0.0, spec);
        }
        let title_bg = Quad {
            position: Vec2::new(self.rect.x, self.rect.y),
            size: Vec2::new(self.rect.width, self.title_height),
            color: style::bg::SECONDARY(),
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&title_bg.to_vertices());
        let title_rect = Rect::new(
            self.rect.x + style::padding::MEDIUM,
            self.rect.y,
            self.rect.width - style::padding::MEDIUM * 2.0,
            self.title_height,
        );
        let mut title_text = crate::ui::text::Text::new_for_render(&self.title)
            .with_font_size(TITLE_FONT_SIZE)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        title_text.update_layout(title_rect, dirty_rect, None);
        renderer.push_parent("section_title".to_string());
        renderer.validate_component("section_title", Some("section"), "SectionTitle");
        title_text.render(renderer, app, vertices, dirty_rect);
        renderer.pop_parent();
        self.content.render(renderer, app, vertices, dirty_rect);
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, dirty_rect: Option<Rect>, app: Option<&App>) {
        self.rect = available_rect;
        let content_rect = Rect::new(
            available_rect.x,
            available_rect.y + self.title_height,
            available_rect.width,
            available_rect.height - self.title_height,
        );
        self.content.update_layout(content_rect, dirty_rect, app);
    }
    
    fn min_size(&self) -> Vec2 {
        let content_size = self.content.min_size();
        Vec2::new(
            content_size.x,
            self.title_height + content_size.y,
        )
    }
}

/// A scrollable list of items
pub struct ScrollableList<T> {
    pub items: Vec<T>,
    pub rect: Rect,
    pub scroll_offset: f32,
    pub item_height: f32,
    pub max_visible_height: Option<f32>,
    pub render_item: fn(&T, &Rect, &mut Renderer, &mut Vec<Vertex>),
    pub selected_index: Option<usize>,
    pub hovered_index: Option<usize>,
}

impl<T> ScrollableList<T> {
    pub fn new(
        items: Vec<T>,
        item_height: f32,
        render_item: fn(&T, &Rect, &mut Renderer, &mut Vec<Vertex>),
    ) -> Self {
        Self {
            items,
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            scroll_offset: 0.0,
            item_height,
            max_visible_height: None,
            render_item,
            selected_index: None,
            hovered_index: None,
        }
    }
    
    pub fn with_max_height(mut self, height: f32) -> Self {
        self.max_visible_height = Some(height);
        self
    }
    
    pub fn with_selected(mut self, index: Option<usize>) -> Self {
        self.selected_index = index;
        self
    }
    
    pub fn with_hovered(mut self, index: Option<usize>) -> Self {
        self.hovered_index = index;
        self
    }
    
    /// Get the total content height
    pub fn content_height(&self) -> f32 {
        self.items.len() as f32 * self.item_height
    }
    
    /// Get the visible height
    pub fn visible_height(&self) -> f32 {
        self.max_visible_height.unwrap_or(self.rect.height).min(self.rect.height)
    }
    
    /// Hit test to find which item was clicked
    pub fn hit_test_item(&self, point: Vec2) -> Option<usize> {
        if !self.rect.contains_point(point) {
            return None;
        }
        
        let local_y = point.y - self.rect.y + self.scroll_offset;
        let index = (local_y / self.item_height) as usize;
        
        if index < self.items.len() {
            Some(index)
        } else {
            None
        }
    }
}

impl<T> Renderable for ScrollableList<T> {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, _dirty_rect: Option<Rect>) {
        let visible_height = self.visible_height();
        
        // Render visible items
        for (i, item) in self.items.iter().enumerate() {
            let item_y = self.rect.y + (i as f32 * self.item_height) - self.scroll_offset;
            
            // Only render if visible
            if item_y + self.item_height >= self.rect.y && item_y < self.rect.y + visible_height {
                let item_rect = Rect::new(
                    self.rect.x,
                    item_y,
                    self.rect.width,
                    self.item_height,
                );
                
                (self.render_item)(item, &item_rect, renderer, vertices);
            }
        }
        
        // Render scrollbar if needed
        if self.content_height() > visible_height {
            let scrollbar_height = (visible_height / self.content_height()) * visible_height;
            let scrollbar_y = self.rect.y + (self.scroll_offset / self.content_height()) * visible_height;
            
            let scrollbar = Quad {
                position: Vec2::new(self.rect.x + self.rect.width - 4.0, scrollbar_y),
                size: Vec2::new(3.0, scrollbar_height),
                color: Vec4::new(0.5, 0.5, 0.5, 0.6),
                corner_radius: 1.5,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&scrollbar.to_vertices());
        }
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        self.rect = available_rect;
    }
    
    fn min_size(&self) -> Vec2 {
        let height = self.max_visible_height.unwrap_or_else(|| self.content_height());
        Vec2::new(200.0, height) // Default min width
    }
}

/// Builder for creating a sidebar with sections
pub struct SidebarBuilder {
    sections: Vec<Box<dyn Renderable>>,
    spacing: f32,
    padding: f32,
}

impl SidebarBuilder {
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
            spacing: style::padding::LARGE,
            padding: style::padding::MEDIUM,
        }
    }
    
    pub fn add_section(mut self, section: Box<dyn Renderable>) -> Self {
        self.sections.push(section);
        self
    }
    
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }
    
    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }
    
    pub fn build(self) -> VStack {
        VStack::new(self.spacing, self.padding)
            .with_children(self.sections)
    }
}

// Example render functions for common item types

/// Render a conversation item
pub fn render_conversation_item(
    conv: &crate::state::Conversation,
    rect: &Rect,
    is_selected: bool,
    is_hovered: bool,
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    const ITEM_FONT_SIZE: f32 = style::font_size::SMALL;
    
    // Background
    let item_color = if is_selected {
        style::highlight::HOVER()
    } else if is_hovered {
        style::bg::TERTIARY()
    } else {
        style::bg::SECONDARY()
    };
    
    let inset_rect = rect.inset(style::padding::TINY);
    let item_bg = Quad {
        position: inset_rect.position(),
        size: inset_rect.size(),
        color: item_color,
        corner_radius: style::corner_radius::MEDIUM,
        bubble_effect: false,
            slider_effect: false,
    };
    vertices.extend_from_slice(&item_bg.to_vertices());
    
    // Title text (truncated)
    let title = if conv.title.len() > 30 {
        format!("{}...", &conv.title[..30])
    } else {
        conv.title.clone()
    };
    
    let text_color = if is_selected {
        style::text::PRIMARY()
    } else if is_hovered {
        Vec4::new(0.95, 0.95, 0.95, 1.0)
    } else {
        style::text::SECONDARY()
    };
    
    // Render title using Text component
    let text_rect = Rect::new(
        inset_rect.x + style::padding::SMALL,
        inset_rect.y,
        inset_rect.width - style::padding::SMALL * 2.0,
        inset_rect.height,
    );
    
    let mut title_text = crate::ui::text::Text::new_for_render(&title)
        .with_font_size(ITEM_FONT_SIZE)
        .with_color(text_color)
        .with_alignment(crate::ui::text::TextAlignment::Left);
    title_text.update_layout(text_rect, None, None);
    renderer.push_parent("conversation_item_title".to_string());
    renderer.validate_component("conversation_item_title", Some("sidebar"), "ConversationItemTitle");
    title_text.render(renderer, app, vertices, None);
    renderer.pop_parent();
}

/// Render a document item
pub fn render_document_item(
    doc_id: &String,
    rect: &Rect,
    is_selected: bool,
    is_hovered: bool,
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    const ITEM_FONT_SIZE: f32 = style::font_size::SMALL;
    
    // Background
    let item_color = if is_selected {
        style::highlight::HOVER()
    } else if is_hovered {
        style::bg::TERTIARY()
    } else {
        style::bg::SECONDARY()
    };
    
    let inset_rect = rect.inset(style::padding::TINY);
    let item_bg = Quad {
        position: inset_rect.position(),
        size: inset_rect.size(),
        color: item_color,
        corner_radius: style::corner_radius::MEDIUM,
        bubble_effect: false,
            slider_effect: false,
    };
    vertices.extend_from_slice(&item_bg.to_vertices());
    
    // Document name (truncated)
    let name = if doc_id.len() > 25 {
        format!("{}...", &doc_id[..25])
    } else {
        doc_id.clone()
    };
    
    let text_color = if is_selected {
        style::text::PRIMARY()
    } else if is_hovered {
        Vec4::new(0.95, 0.95, 0.95, 1.0)
    } else {
        style::text::SECONDARY()
    };
    
    // Render document name using Text component
    let text_rect = Rect::new(
        inset_rect.x + style::padding::SMALL,
        inset_rect.y,
        inset_rect.width - style::padding::SMALL * 2.0,
        inset_rect.height,
    );
    
    let mut name_text = crate::ui::text::Text::new_for_render(&name)
        .with_font_size(ITEM_FONT_SIZE)
        .with_color(text_color)
        .with_alignment(crate::ui::text::TextAlignment::Left);
    name_text.update_layout(text_rect, None, None);
    renderer.push_parent("document_item_name".to_string());
    renderer.validate_component("document_item_name", Some("sidebar"), "DocumentItemName");
    name_text.render(renderer, app, vertices, None);
    renderer.pop_parent();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scrollable_list_hit_test() {
        let list: ScrollableList<String> = ScrollableList::new(
            vec!["Item 1".into(), "Item 2".into(), "Item 3".into()],
            40.0,
            |_, _, _, _| {},
        );
        
        let mut list = list;
        list.update_layout(Rect::new(0.0, 0.0, 200.0, 120.0), None, None);
        
        // Click on first item
        assert_eq!(list.hit_test_item(Vec2::new(100.0, 20.0)), Some(0));
        
        // Click on second item
        assert_eq!(list.hit_test_item(Vec2::new(100.0, 50.0)), Some(1));
        
        // Click outside
        assert_eq!(list.hit_test_item(Vec2::new(300.0, 50.0)), None);
    }
    
    #[test]
    fn test_section_layout() {
        struct MockContent {
            rect: Rect,
        }
        
        impl Renderable for MockContent {
            fn render(&self, _: &mut Renderer, _: &App, _: &mut Vec<Vertex>, _: Option<Rect>) {}
            fn bounds(&self) -> Rect { self.rect }
            fn update_layout(&mut self, available_rect: Rect, _: Option<Rect>, _: Option<&App>) { self.rect = available_rect; }
            fn min_size(&self) -> Vec2 { Vec2::new(200.0, 100.0) }
        }
        
        let mut section = Section::new(
            "Test Section",
            Box::new(MockContent { rect: Rect::new(0.0, 0.0, 0.0, 0.0) }),
        );
        
        section.update_layout(Rect::new(0.0, 0.0, 200.0, 200.0), None, None);
        
        // Content should start below title
        assert_eq!(section.content.bounds().y, section.title_height);
        
        // Content height should be total height minus title
        assert_eq!(section.content.bounds().height, 200.0 - section.title_height);
    }
}

