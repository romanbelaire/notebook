use glam::Vec2;
use crate::ui::core::Rect;
use crate::ui::components::{Renderable, VStack};
use crate::utils::animation::{SpringAnimation, AnimationPreset};
use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use crate::ui::{Text, TextAlignment};
use crate::ui::style;
use crate::ui::icons::icon_names;

pub struct Dropdown {
    pub anchor_rect: Rect,  // Button bounds for relative positioning
    pub button_size: Vec2,
    pub is_open: bool,
    pub items: Vec<DropdownItem>,
    pub selected_index: Option<usize>,
    pub menu_rect: Rect,  // Calculated menu position
    pub open_animation: SpringAnimation,  // For smooth open/close
    pub menu_vstack: Option<VStack>,  // For menu item layout
}

#[derive(Clone, Debug)]
pub struct DropdownItem {
    pub id: Option<i32>,  // None for "All papers"
    pub label: String,
}

impl Dropdown {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        Self {
            anchor_rect: Rect::from_pos_size(position, size),
            button_size: size,
            is_open: false,
            items: Vec::new(),
            selected_index: None,
            menu_rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            open_animation: SpringAnimation::with_preset(0.0, AnimationPreset::Snappy),
            menu_vstack: None,
        }
    }

    pub fn contains(&self, p: Vec2) -> bool {
        self.anchor_rect.contains_point(p)
    }

    pub fn contains_menu(&self, p: Vec2) -> bool {
        if !self.is_open || self.open_animation.value < 0.1 {
            return false;
        }
        self.menu_rect.contains_point(p)
    }

    pub fn get_menu_item_at(&self, p: Vec2) -> Option<usize> {
        if !self.is_open || self.open_animation.value < 0.1 {
            return None;
        }
        
        let menu_padding = 10.0;
        let item_height = 30.0;
        let menu_content_y = self.menu_rect.y + menu_padding;
        let rel_y = p.y - menu_content_y;
        
        if rel_y < 0.0 {
            return None;
        }
        
        // Check if clicking on "Create new collection" button
        let items_height = self.items.len() as f32 * item_height;
        if rel_y > items_height + 5.0 {
            return None;  // This is handled separately
        }
        
        let index = (rel_y / item_height) as usize;
        if index < self.items.len() {
            Some(index)
        } else {
            None
        }
    }

    pub fn toggle(&mut self) {
        self.is_open = !self.is_open;
        self.open_animation.target = if self.is_open { 1.0 } else { 0.0 };
        // Always update layout when toggling to ensure VStack is built
        if self.is_open {
            self.update_layout();
        }
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.open_animation.target = 0.0;
    }

    pub fn select(&mut self, index: usize) {
        if index < self.items.len() {
            self.selected_index = Some(index);
            self.close();
        }
    }

    pub fn get_selected_id(&self) -> Option<i32> {
        self.selected_index.and_then(|idx| self.items.get(idx).and_then(|item| item.id))
    }

    pub fn set_selected_by_id(&mut self, collection_id: Option<i32>) {
        if let Some(id) = collection_id {
            self.selected_index = self.items.iter().position(|item| item.id == Some(id));
        } else {
            // "All papers" is selected (id is None)
            self.selected_index = self.items.iter().position(|item| item.id.is_none());
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.open_animation.update(dt);
    }

    pub fn update_layout(&mut self) {
        if !self.is_open {
            return;
        }

        use crate::ui::core::layout;
        
        let menu_padding = style::padding::SMALL;
        let item_height = 30.0;
        let menu_spacing = 5.0;
        let create_button_height = 30.0;
        
        // Calculate menu height
        let items_height = self.items.len() as f32 * item_height;
        let menu_height = items_height + menu_padding * 2.0 + menu_spacing + create_button_height;
        
        // Calculate menu width (match button width or use minimum)
        let menu_width = self.button_size.x.max(200.0);
        
        // Position menu above button with spacing
        let menu_spacing_from_button = 8.0;
        let menu_y = self.anchor_rect.y - menu_height - menu_spacing_from_button;
        
        // Align menu with button (left-aligned)
        let menu_x = self.anchor_rect.x;
        
        self.menu_rect = Rect::new(menu_x, menu_y, menu_width, menu_height);
        
        // Build or update VStack for menu items
        // Only rebuild if items changed or VStack doesn't exist
        let needs_rebuild = self.menu_vstack.is_none() || 
            self.menu_vstack.as_ref().map(|v| v.children.len()) != Some(self.items.len());
        
        if needs_rebuild {
            let mut vstack = VStack::new(0.0, 0.0); // No spacing between items, no padding (we handle it)
            
            // Add menu items as Text components
            for item in &self.items {
                vstack.add_text_styled(
                    &item.label,
                    style::font_size::NORMAL,
                    style::text::PRIMARY,
                    TextAlignment::Left,
                );
            }
            
            self.menu_vstack = Some(vstack);
        }
        
        // Update VStack layout - set each item to fixed height for consistent spacing
        let content_rect = self.menu_rect.inset(menu_padding);
        if let Some(ref mut stack) = self.menu_vstack {
            // Manually layout each child with fixed height
            let mut current_y = content_rect.y;
            for child in &mut stack.children {
                let child_rect = Rect::new(
                    content_rect.x,
                    current_y,
                    content_rect.width,
                    item_height, // Fixed height for each item
                );
                child.update_layout(child_rect, None, None);
                current_y += item_height;
            }
        }
    }
}

impl Renderable for Dropdown {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        // Render menu if animation value is significant (allows closing animation to complete)
        // Continue rendering even when is_open is false if animation hasn't finished
        if self.open_animation.value > 0.01 {
            let opacity = self.open_animation.value;
            
            // Calculate slide-up animation offset
            // Menu starts at button position and slides up to final position
            let slide_distance = self.menu_rect.height + 8.0; // Distance to slide up
            let slide_offset = slide_distance * (1.0 - opacity); // More offset when opacity is lower
            let animated_y = self.menu_rect.y + slide_offset;
            
            // Create animated menu rect
            let animated_menu_rect = Rect::new(
                self.menu_rect.x,
                animated_y,
                self.menu_rect.width,
                self.menu_rect.height,
            );
            
            // Menu background with animation opacity
            let mut menu_bg_color = style::bg::SECONDARY;
            menu_bg_color.w *= opacity;
            
            let menu_bg = Quad {
                position: animated_menu_rect.position(),
                size: animated_menu_rect.size(),
                color: menu_bg_color,
                corner_radius: style::corner_radius::MEDIUM,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&menu_bg.to_vertices());
            
            // Render highlights for selected items first (behind text)
            // Adjust positions by slide offset
            let menu_padding = style::padding::SMALL;
            let item_height = 30.0;
            
            if let Some(ref vstack) = self.menu_vstack {
                for (index, child) in vstack.children.iter().enumerate() {
                    if index < self.items.len() {
                        let is_selected = self.selected_index == Some(index);
                        if is_selected {
                            let child_bounds = child.bounds();
                            // Adjust highlight position by slide offset
                            let animated_highlight_rect = Rect::new(
                                child_bounds.x,
                                child_bounds.y + slide_offset,
                                child_bounds.width,
                                child_bounds.height,
                            );
                            let mut highlight_color = style::highlight::SELECTION;
                            highlight_color.w *= opacity;
                            let highlight_bg = Quad {
                                position: animated_highlight_rect.position(),
                                size: animated_highlight_rect.size(),
                                color: highlight_color,
                                corner_radius: style::corner_radius::SMALL,
                                bubble_effect: false,
                slider_effect: false,
                            };
                            vertices.extend_from_slice(&highlight_bg.to_vertices());
                        }
                    }
                }
            }
            
            // Render menu items using VStack (properly aligned)
            // Adjust positions by slide offset
            if let Some(ref vstack) = self.menu_vstack {
                // Push dropdown menu as parent
                renderer.push_parent("dropdown_menu".to_string());
                renderer.validate_component("dropdown_menu", None, "DropdownMenu");
                
                // Render VStack children with animated positions
                for (index, child) in vstack.children.iter().enumerate() {
                    if index < self.items.len() {
                        let child_bounds = child.bounds();
                        // Temporarily adjust child position for rendering
                        // We need to create a modified child or adjust rendering position
                        // Since we can't mutate, we'll need to adjust the rendering context
                        // For now, let's update the child's layout temporarily
                        let animated_child_rect = Rect::new(
                            child_bounds.x,
                            child_bounds.y + slide_offset,
                            child_bounds.width,
                            child_bounds.height,
                        );
                        
                        // Create a temporary text component with animated position
                        use crate::ui::text::Text;
                        let mut item_text = Text::new_for_render(&self.items[index].label)
                            .with_font_size(style::font_size::NORMAL)
                            .with_color(style::text::PRIMARY)
                            .with_alignment(TextAlignment::Left);
                        item_text.update_layout(animated_child_rect, dirty_rect, None);
                        
                        let component_id = format!("dropdown_item_{}", index);
                        renderer.push_parent(component_id.clone());
                        renderer.validate_component(&component_id, Some("dropdown_menu"), "DropdownItem");
                        item_text.render(renderer, app, vertices, dirty_rect);
                        renderer.pop_parent();
                    }
                }
                
                renderer.pop_parent();
            }
            
            // "Create new collection" button at bottom (with animated position)
            let menu_spacing = 5.0;
            let create_button_y = animated_menu_rect.y + menu_padding + (self.items.len() as f32 * item_height) + menu_spacing;
            let create_button_rect = Rect::new(
                animated_menu_rect.x + menu_padding,
                create_button_y,
                animated_menu_rect.width - menu_padding * 2.0,
                30.0,
            );
            
            let mut button_color = style::button::SECONDARY;
            button_color.w *= opacity;
            let create_button_bg = Quad {
                position: create_button_rect.position(),
                size: create_button_rect.size(),
                color: button_color,
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&create_button_bg.to_vertices());
            
            // Plus icon + text
            let plus_icon_size = 14.0;
            let plus_icon_pos = Vec2::new(
                create_button_rect.x + 8.0,
                create_button_rect.y + create_button_rect.height / 2.0 - plus_icon_size / 2.0,
            );
            let mut icon_color = style::text::PRIMARY;
            icon_color.w *= opacity;
            renderer.queue_icon(
                icon_names::PLUS,
                plus_icon_pos,
                plus_icon_size,
                icon_color,
            );
            
            let create_text_rect = Rect::new(
                create_button_rect.x + 25.0,
                create_button_rect.y,
                create_button_rect.width - 25.0,
                create_button_rect.height,
            );
            let mut create_text = Text::new_for_render("Create new collection")
                .with_font_size(style::font_size::SMALL)
                .with_color(icon_color)
                .with_alignment(TextAlignment::Left);
            create_text.update_layout(create_text_rect, dirty_rect, None);
            
            renderer.push_parent("dropdown_create".to_string());
            renderer.validate_component("dropdown_create", None, "DropdownCreate");
            create_text.render(renderer, app, vertices, dirty_rect);
            renderer.pop_parent();
        }
    }
    
    fn bounds(&self) -> Rect {
        self.anchor_rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        if available_rect.width > 0.0 && available_rect.height > 0.0 {
            self.anchor_rect = available_rect;
        }
        self.update_layout();
    }
    
    fn min_size(&self) -> Vec2 {
        self.button_size
    }
}

