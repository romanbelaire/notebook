use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::core::{Rect, text_input_render};
use crate::ui::{Text, TextAlignment};
use crate::ui::components::Renderable;

pub fn render_library(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if app.ui_state.active_tab != Tab::Library {
        return;
    }
    
    // Set parent context for library components
    // Note: "library" is already validated by the renderer as a RenderableComponent
    // We just need to push it as parent for child components
    renderer.push_parent("library".to_string());
    
    if let Some(ref library) = app.library_window {
        // Background
        let bg = Quad {
            position: library.position,
            size: library.size,
            color: style::bg::PRIMARY,
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&bg.to_vertices());

        const PADDING: f32 = 10.0;
        const LEFT_PANEL_WIDTH: f32 = 300.0;
        const SEARCH_HEIGHT: f32 = 40.0;
        const COLLECTION_ITEM_HEIGHT: f32 = 35.0;
        const PAPER_ITEM_HEIGHT: f32 = 50.0;

        // Left panel (Collections)
        let left_panel_rect = Rect::new(
            library.position.x + PADDING,
            library.position.y + PADDING,
            LEFT_PANEL_WIDTH - PADDING * 2.0,
            library.size.y - PADDING * 2.0,
        );
        
        let left_panel_bg = Quad {
            position: left_panel_rect.position(),
            size: left_panel_rect.size(),
            color: style::bg::SECONDARY,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&left_panel_bg.to_vertices());

        // Use vertical stack for left panel contents: button, input (conditional), list
        let button_height = library.new_collection_button.size.y;
        let input_height = if library.is_creating_collection { library.new_collection_input.size.y } else { 0.0 };
        let list_start_y = left_panel_rect.y + button_height + (if input_height > 0.0 { input_height + PADDING } else { 0.0 });
        
        // New Collection button
        let button_rect = Rect::new(
            left_panel_rect.x + PADDING,
            left_panel_rect.y + PADDING,
            left_panel_rect.width - PADDING * 2.0,
            button_height,
        );
        let button_bg = Quad {
            position: button_rect.position(),
            size: button_rect.size(),
            color: match library.new_collection_button.state {
                crate::ui::ButtonState::Pressed => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.8),
                crate::ui::ButtonState::Hover => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.9),
                crate::ui::ButtonState::Normal => style::button::PRIMARY,
            },
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&button_bg.to_vertices());
        
        let text_width = renderer.measure_text(&library.new_collection_button.label, style::font_size::NORMAL).x;
        let _text_pos = crate::ui::core::text::center_aligned(&button_rect, text_width, style::font_size::NORMAL);
        // Render new collection button label using Text component
        let button_text_rect = Rect::new(
            button_rect.x,
            button_rect.y,
            button_rect.width,
            button_rect.height,
        );
        
        let mut button_text = crate::ui::text::Text::new_for_render(&library.new_collection_button.label)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY)
            .with_alignment(crate::ui::text::TextAlignment::Center);
        button_text.update_layout(button_text_rect, None, None);
        
        renderer.push_parent("library_new_collection_button".to_string());
        renderer.validate_component("library_new_collection_button", Some("library"), "NewCollectionButton");
        button_text.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // New collection input (if creating)
        if library.is_creating_collection {
            let input_rect = Rect::new(
                left_panel_rect.x + PADDING,
                button_rect.bottom() + PADDING,
                left_panel_rect.width - PADDING * 2.0,
                input_height,
            );
            let input_bg = Quad {
                position: input_rect.position(),
                size: input_rect.size(),
                color: if library.new_collection_input.focused {
                    style::bg::INPUT_FOCUSED
                } else {
                    style::bg::INPUT
                },
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&input_bg.to_vertices());
            
            // Render new collection input text using Text component
            let input_text_rect = Rect::new(
                input_rect.x + style::padding::SMALL,
                input_rect.y,
                input_rect.width - style::padding::SMALL * 2.0,
                input_rect.height,
            );
            
            let mut input_text = crate::ui::text::Text::new_for_render(&library.new_collection_input.text)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            input_text.update_layout(input_text_rect, None, None);
            
            renderer.push_parent("library_new_collection_input".to_string());
            renderer.validate_component("library_new_collection_input", Some("library"), "NewCollectionInput");
            input_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }

        // Collections list
        let collections_start_y = list_start_y;
        let scroll_offset = library.collections_list.scroll_offset;
        let mut current_y = collections_start_y - scroll_offset + 10.0;

        // "All papers" option
        let all_papers_rect = Rect::new(
            left_panel_rect.x,
            current_y,
            left_panel_rect.width,
            COLLECTION_ITEM_HEIGHT,
        );
        
        // Highlight if no collection is selected
        if library.selected_collection_id.is_none() {
            let highlight = Quad {
                position: all_papers_rect.position(),
                size: all_papers_rect.size(),
                color: style::highlight::SELECTION,
                corner_radius: 0.0,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&highlight.to_vertices());
        }
        
        // "All papers" text - use Text component
        let all_papers_text_rect = Rect::new(
            all_papers_rect.x + PADDING,
            current_y + 10.0,
            all_papers_rect.width - PADDING * 2.0,
            COLLECTION_ITEM_HEIGHT - 20.0,
        );
        renderer.push_parent("library_all_papers".to_string());
        renderer.validate_component("library_all_papers", Some("library"), "AllPapers");
        let mut all_papers_text = Text::new_for_render("All papers")
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Left);
        all_papers_text.update_layout(all_papers_text_rect, None, None);
        all_papers_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
        current_y += COLLECTION_ITEM_HEIGHT;

        // Render collections
        for (idx, collection) in library.collections.iter().enumerate() {
            if current_y + COLLECTION_ITEM_HEIGHT < collections_start_y {
                current_y += COLLECTION_ITEM_HEIGHT;
                continue;
            }
            if current_y > collections_start_y + library.collections_list.size.y {
                break;
            }

            let collection_rect = Rect::new(
                left_panel_rect.x,
                current_y,
                left_panel_rect.width,
                COLLECTION_ITEM_HEIGHT,
            );

            // Highlight selected collection
            if library.selected_collection_id == Some(collection.id) {
                let highlight = Quad {
                    position: collection_rect.position(),
                    size: collection_rect.size(),
                    color: style::highlight::SELECTION,
                    corner_radius: 0.0,
                    bubble_effect: false,
            slider_effect: false,
                };
                vertices.extend_from_slice(&highlight.to_vertices());
            }

            // Collection name text - use Text component
            let display_text = format!("{} ({})", collection.name, collection.paper_count);
            let collection_text_rect = Rect::new(
                collection_rect.x + PADDING,
                current_y + 10.0,
                collection_rect.width - PADDING * 2.0,
                COLLECTION_ITEM_HEIGHT - 20.0,
            );
            let collection_parent_id = format!("library_collection_{}", idx);
            renderer.push_parent(collection_parent_id.clone());
            renderer.validate_component(&collection_parent_id, Some("library"), "Collection");
            let mut collection_text = Text::new_for_render(&display_text)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY)
                .with_alignment(TextAlignment::Left);
            collection_text.update_layout(collection_text_rect, None, None);
            collection_text.render(renderer, app, vertices, None);
            renderer.pop_parent();

            current_y += COLLECTION_ITEM_HEIGHT;
        }

        // Right panel (Papers)
        let right_panel_x = library.position.x + LEFT_PANEL_WIDTH;
        let right_panel_width = library.size.x - LEFT_PANEL_WIDTH;
        let _right_panel_rect = Rect::new(
            right_panel_x + PADDING,
            library.position.y + PADDING,
            right_panel_width - PADDING * 2.0,
            library.size.y - PADDING * 2.0,
        );

        // Use vertical stack for right panel: search input, papers list
        // Search input - use standard text input rendering
        text_input_render::render_text_input(
            renderer,
            &library.search_input,
            app,
            vertices,
            None, // Use default font size
            None, // Use default padding
            None, // Use default corner radius
            false,
        );

        // Papers list
        let papers_start_y = library.search_input.position.y + library.search_input.size.y + PADDING;
        let papers_scroll_offset = library.papers_list.scroll_offset;
        let mut paper_y = papers_start_y - papers_scroll_offset + 10.0;

        if library.filtered_papers.is_empty() {
            // Empty state text - use Text component
            let empty_text = if library.search_query.is_empty() {
                "No papers found"
            } else {
                "No papers match your search"
            };
            let empty_text_rect = Rect::new(
                library.papers_list.position.x + PADDING,
                papers_start_y + PADDING,
                library.papers_list.size.x - PADDING * 2.0,
                30.0,
            );
            renderer.push_parent("library_empty_state".to_string());
            renderer.validate_component("library_empty_state", Some("library"), "EmptyState");
            let mut empty_text_component = Text::new_for_render(empty_text)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::SECONDARY)
                .with_alignment(TextAlignment::Left);
            empty_text_component.update_layout(empty_text_rect, None, None);
            empty_text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();
        } else {
            // Delete button (if papers are selected)
            if !library.selected_papers.is_empty() {
                let delete_button_rect = Rect::from_pos_size(
                    library.delete_button.position,
                    library.delete_button.size,
                );
                let delete_bg = Quad {
                    position: delete_button_rect.position(),
                    size: delete_button_rect.size(),
                    color: if library.delete_confirm {
                        style::button::DANGER
                    } else {
                        match library.delete_button.state {
                            crate::ui::ButtonState::Pressed => style::button::SECONDARY * Vec4::new(1.0, 1.0, 1.0, 0.8),
                            crate::ui::ButtonState::Hover => style::button::SECONDARY * Vec4::new(1.0, 1.0, 1.0, 0.9),
                            crate::ui::ButtonState::Normal => style::button::SECONDARY,
                        }
                    },
                    corner_radius: style::corner_radius::SMALL,
                    bubble_effect: false,
            slider_effect: false,
                };
                vertices.extend_from_slice(&delete_bg.to_vertices());
                
                let delete_label = if library.delete_confirm {
                    "Confirm Delete"
                } else {
                    "Delete"
                };
                let mut delete_text = Text::new_for_render(delete_label)
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(style::text::PRIMARY)
                    .with_alignment(TextAlignment::Center);
                delete_text.update_layout(delete_button_rect, None, None);
                renderer.push_parent("library_delete_button".to_string());
                renderer.validate_component("library_delete_button", Some("library"), "DeleteButton");
                delete_text.render(renderer, app, vertices, None);
                renderer.pop_parent();
            }

            for (paper_idx, paper) in library.filtered_papers.iter().enumerate() {
                if paper_y + PAPER_ITEM_HEIGHT < papers_start_y {
                    paper_y += PAPER_ITEM_HEIGHT;
                    continue;
                }
                if paper_y > papers_start_y + library.papers_list.size.y {
                    break;
                }

                let paper_rect = Rect::new(
                    library.papers_list.position.x,
                    paper_y,
                    library.papers_list.size.x,
                    PAPER_ITEM_HEIGHT,
                );

                // Paper item background
                let paper_bg = Quad {
                    position: paper_rect.position(),
                    size: paper_rect.size(),
                    color: style::bg::SECONDARY,
                    corner_radius: style::corner_radius::SMALL,
                    bubble_effect: false,
            slider_effect: false,
                };
                vertices.extend_from_slice(&paper_bg.to_vertices());

                // Checkbox
                const CHECKBOX_SIZE: f32 = 16.0;
                let checkbox_pos = Vec2::new(
                    paper_rect.x + PADDING,
                    paper_y + (PAPER_ITEM_HEIGHT - CHECKBOX_SIZE) / 2.0,
                );
                let checkbox_rect = Rect::new(
                    checkbox_pos.x,
                    checkbox_pos.y,
                    CHECKBOX_SIZE,
                    CHECKBOX_SIZE,
                );
                
                // Checkbox border
                let checkbox_border = Quad {
                    position: checkbox_rect.position(),
                    size: checkbox_rect.size(),
                    color: style::border::DEFAULT,
                    corner_radius: 2.0,
                    bubble_effect: false,
            slider_effect: false,
                };
                vertices.extend_from_slice(&checkbox_border.to_vertices());
                
                // Checkbox fill if selected
                if library.is_paper_selected(paper.id) {
                    let checkbox_fill = Quad {
                        position: checkbox_rect.position() + Vec2::new(2.0, 2.0),
                        size: checkbox_rect.size() - Vec2::new(4.0, 4.0),
                        color: style::button::PRIMARY,
                        corner_radius: 1.0,
                        bubble_effect: false,
            slider_effect: false,
                    };
                    vertices.extend_from_slice(&checkbox_fill.to_vertices());
                }

                // Paper title or filename - use Text component
                let display_name = paper.title.as_ref().unwrap_or(&paper.filename);
                let title_rect = Rect::new(
                    paper_rect.x + PADDING + CHECKBOX_SIZE + PADDING,
                    paper_y + 8.0,
                    paper_rect.width - PADDING * 3.0 - CHECKBOX_SIZE,
                    20.0,
                );
                let paper_title_id = format!("library_paper_title_{}", paper_y as i32);
                renderer.push_parent(paper_title_id.clone());
                renderer.validate_component(&paper_title_id, Some("library"), "PaperTitle");
                let mut title_text = Text::new_for_render(display_name)
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(style::text::PRIMARY)
                    .with_alignment(TextAlignment::Left);
                title_text.update_layout(title_rect, None, None);
                title_text.render(renderer, app, vertices, None);
                renderer.pop_parent();

                // Paper metadata (authors, year) - use Text component
                let mut metadata = String::new();
                if let Some(ref authors) = paper.authors {
                    metadata.push_str(authors);
                }
                if let Some(year) = paper.year {
                    if !metadata.is_empty() {
                        metadata.push_str(", ");
                    }
                    metadata.push_str(&year.to_string());
                }
                if !metadata.is_empty() {
                    let metadata_rect = Rect::new(
                        paper_rect.x + PADDING + CHECKBOX_SIZE + PADDING,
                        paper_y + 28.0,
                        paper_rect.width - PADDING * 3.0 - CHECKBOX_SIZE,
                        20.0,
                    );
                    let paper_metadata_id = format!("library_paper_metadata_{}", paper_y as i32);
                    renderer.push_parent(paper_metadata_id.clone());
                    renderer.validate_component(&paper_metadata_id, Some("library"), "PaperMetadata");
                    let mut metadata_text = Text::new_for_render(&metadata)
                        .with_font_size(style::font_size::SMALL)
                        .with_color(style::text::SECONDARY)
                        .with_alignment(TextAlignment::Left);
                    metadata_text.update_layout(metadata_rect, None, None);
                    metadata_text.render(renderer, app, vertices, None);
                    renderer.pop_parent();
                }

                paper_y += PAPER_ITEM_HEIGHT;
            }
        }
    }
    
    // Pop library parent
    renderer.pop_parent();
}

