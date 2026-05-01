use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::persistence::DocumentPersistence;
use crate::ui::style;
use crate::ui::core::{Rect, container, layout};
use crate::ui::{Text, TextAlignment};
use crate::ui::components::Renderable;

pub fn render_sidebar_content(renderer: &mut Renderer, app: &App, _vertices: &mut Vec<Vertex>) {
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
    // Quads/highlights stay on SidebarChrome; every Text::render below temporarily switches to
    // HudChrome so sidebar labels composite above ConstellationText (same idea as context menus).
    // Use current_width instead of is_open to allow content to animate with sidebar
    // Only render if sidebar has enough width to show any content
    const MIN_WIDTH_FOR_CONTENT: f32 = 10.0;
    if app.sidebar.current_width < MIN_WIDTH_FOR_CONTENT {
        return;
    }
    
    // Note: "sidebar_content" is already validated and pushed by SidebarContentComponent
    // before calling this function. We don't need to push/pop it here.
    // The parent stack already has "sidebar_content" on it.
    
    const FONT_SIZE: f32 = style::font_size::MEDIUM;
    const TITLE_FONT_SIZE: f32 = style::font_size::NORMAL;
    const ITEM_TITLE_FONT_SIZE: f32 = style::font_size::SMALL;
    
    let padding = style::padding::MEDIUM;
    let item_height = 40.0;
    
    // Create sidebar rect (position already includes header offset)
    // Use current_width so content animates smoothly with sidebar
    // Calculate translation offset: as sidebar collapses, content should translate left
    // When fully open: offset = 0, when fully closed: offset = -(OPEN_WIDTH - CLOSED_WIDTH)
    let open_width = crate::ui::sidebar::SidebarWindow::OPEN_WIDTH;
    let closed_width = 1.0; // CLOSED_WIDTH
    let width_delta = open_width - app.sidebar.current_width;
    let translation_offset = -width_delta; // Negative = move left as sidebar collapses
    
    let sidebar_rect = Rect::new(
        app.sidebar.position.x + translation_offset,
        app.sidebar.position.y,  // Already positioned after header (y=60)
        app.sidebar.current_width,
        app.sidebar.height,  // Already excludes header
    );
    
    // Build section stack: three separate sections (VStack order).
    // Section 0 = Conversations, 1 = Documents, 2 = Insights (pinned shards).
    // Insights is its own section, not inside the chats menu.
    let mut stack = container::SectionStack::new(style::padding::LARGE);
    
    // Section 0: Conversations
    let mut conversations_section = container::Section::new("Conversations".to_string(), item_height);
    conversations_section.title_height = 40.0;
    conversations_section.item_count = app.chat_state.conversations.len();
    conversations_section.max_content_height = Some(250.0);
    conversations_section.scroll_offset = app.sidebar.conversations_list.scroll_view.scroll_offset;
    stack.add_section(conversations_section);
    
    // Section 1: Documents
    let document_count = DocumentPersistence::list_documents().map(|docs| docs.len()).unwrap_or(0);
    let mut documents_section = container::Section::new("Documents".to_string(), item_height);
    documents_section.title_height = 40.0;
    documents_section.item_count = document_count;
    documents_section.max_content_height = Some(250.0);
    documents_section.scroll_offset = app.sidebar.documents_list.scroll_view.scroll_offset;
    stack.add_section(documents_section);
    
    // Section 2: Collections
    let mut collections_section = container::Section::new("Collections".to_string(), item_height);
    collections_section.title_height = 40.0;
    collections_section.item_count = app.library_window.as_ref().map(|w| w.collections.len()).unwrap_or(0);
    collections_section.max_content_height = Some(250.0);
    collections_section.scroll_offset = app.sidebar.collections_list.scroll_view.scroll_offset;
    stack.add_section(collections_section);

    // Section 3: Insights (pinned shards) – item_height 35 to match sidebar hit-test
    const INSIGHTS_ITEM_HEIGHT: f32 = 35.0;
    let mut insights_section = container::Section::new("Insights".to_string(), INSIGHTS_ITEM_HEIGHT);
    insights_section.title_height = 40.0;
    insights_section.item_count = app.insights_state.insights.len();
    insights_section.scroll_offset = app.sidebar.insights_panel.insights_list.scroll_view.scroll_offset;
    insights_section.max_content_height = None; // Show all
    stack.add_section(insights_section);
    
    // Get layout
    let layout = stack.layout(&sidebar_rect);
    
    // Render each section
    for (section_idx, y_offset) in layout {
        let section = &stack.sections[section_idx];
        // Render section title and content based on type
        match section_idx {
            0 => render_conversations_section(app, &sidebar_rect, y_offset, section, renderer, padding, item_height),
            1 => render_documents_section(app, &sidebar_rect, y_offset, section, renderer, padding, item_height),
            2 => render_collections_section(app, &sidebar_rect, y_offset, section, renderer, padding, item_height),
            3 => render_insights_section(app, &sidebar_rect, y_offset, section, renderer, padding),
            _ => {}
        }
    }

    // Settings button: only shown on Chat tab when constellation is active
    use crate::ui::tab_bar::Tab;
    use crate::ui::icons::icon_names;
    if app.ui_state.active_tab == Tab::Chat && app.graph_state.constellation_view_active() {
        render_settings_button(app, &sidebar_rect, renderer);
        if app.sidebar.settings_panel_open {
            render_settings_panel(app, &sidebar_rect, renderer);
        }
    }
}

fn render_conversations_section(
    app: &App,
    sidebar_rect: &Rect,
    y_offset: f32,
    section: &container::Section,
    renderer: &mut Renderer,
    padding: f32,
    item_height: f32,
) {
    // Calculate text animation parameters
    let min_width_for_text = 100.0;
    let text_opacity = if app.sidebar.current_width < min_width_for_text {
        (app.sidebar.current_width / min_width_for_text).max(0.0)
    } else {
        1.0
    };
    let open_width = crate::ui::sidebar::SidebarWindow::OPEN_WIDTH;
    let text_x_offset = if app.sidebar.current_width < open_width {
        (app.sidebar.current_width / open_width) * padding
    } else {
        padding
    };
    const TITLE_FONT_SIZE: f32 = style::font_size::NORMAL;
    const ITEM_TITLE_FONT_SIZE: f32 = style::font_size::SMALL;
    const FONT_SIZE: f32 = style::font_size::MEDIUM;
    
    // Render title (unclipped - title is outside scrollable area)
    let title_rect = section.title_rect(sidebar_rect, y_offset);
    // Title position calculated but not stored (used inline in rect calculations)
    // Use horizontal stack to position title and button
    let button_size = Vec2::new(30.0, 30.0);
    let title_text_width = renderer.measure_text("Conversations", TITLE_FONT_SIZE).x;
    let title_container = Rect::new(
        title_rect.x,
        title_rect.y,
        title_rect.width,
        title_rect.height,
    );
    
    let title_rects = layout::stack_horizontal(
        &title_container,
        &[title_text_width, button_size.x],
        padding,
        padding,
    );
    
    // Apply opacity to title text
    let mut title_color = style::text::PRIMARY();
    title_color.w *= text_opacity;
    // Title - use Text component (no scissor, it's not in scrollable area)
    let title_text_rect = Rect::new(
        title_rects[0].x,
        title_rect.y + (title_rect.height - TITLE_FONT_SIZE * 1.2) / 2.0,
        title_rects[0].width,
        TITLE_FONT_SIZE * 1.2,
    );
    // Validate title BEFORE pushing it (so it's in hierarchy when Text validates)
    renderer.validate_component("sidebar_conversations_title", Some("sidebar_content"), "SectionTitle");
    renderer.push_parent("sidebar_conversations_title".to_string());
    let mut title_text = Text::new_for_render("Conversations")
        .with_font_size(TITLE_FONT_SIZE)
        .with_color(title_color)
        .with_alignment(TextAlignment::Left)
        .with_scissor(None);
    title_text.update_layout(title_text_rect, None, None);
    let mut text_vertices = Vec::new();
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
    title_text.render(renderer, app, &mut text_vertices, None);
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
    if !text_vertices.is_empty() {
        renderer.add_vertices(&text_vertices, None);
    }
    renderer.pop_parent();
    
    // New conversation button (right side of title area)
    let button_rect = Rect::new(
        title_rects[1].x,
        title_rect.y + (title_rect.height - button_size.y) / 2.0,
        button_size.x,
        button_size.y,
    );
    let new_conv_color = match app.sidebar.new_conversation_button.state {
        crate::ui::ButtonState::Pressed => style::button::PRIMARY_ACTIVE(),
        crate::ui::ButtonState::Hover => style::button::PRIMARY_HOVER(),
        crate::ui::ButtonState::Normal => style::button::PRIMARY(),
    };
    let new_conv_bg = Quad {
        position: button_rect.position(),
        size: button_rect.size(),
        color: new_conv_color,
        corner_radius: style::corner_radius::MEDIUM,
        bubble_effect: false,
        slider_effect: false,
    };
    // Button is in title area - use sidebar_rect for scissor to ensure it renders
    renderer.add_quad(&new_conv_bg, Some(sidebar_rect));
    // Button text - use Text component (no scissor, it's in title area)
    renderer.push_parent("sidebar_conversations_button".to_string());
    renderer.validate_component("sidebar_conversations_button", None, "Button");
    let mut plus_text = Text::new_for_render("+")
        .with_font_size(FONT_SIZE)
        .with_color(style::text::PRIMARY())
        .with_alignment(TextAlignment::Center)
        .with_scissor(None);
    plus_text.update_layout(button_rect, None, None);
    let mut text_vertices = Vec::new();
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
    plus_text.render(renderer, app, &mut text_vertices, None);
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
    if !text_vertices.is_empty() {
        renderer.add_vertices(&text_vertices, None);
    }
    renderer.pop_parent();
    
    // Get content rect for clipping scrolled content
    let content_rect = section.content_rect(sidebar_rect, y_offset);
    
    // Render unified highlight bar for conversations list (hover)
    if app.sidebar.conversations_list.scroll_view.highlight_bar_visible {
        let highlight_y = app.sidebar.conversations_list.scroll_view.highlight_bar_y;
        let highlight_rect = Rect::new(
            content_rect.x + padding,
            highlight_y,
            content_rect.width - padding * 2.0,
            item_height,
        );
        let highlight_bg = Quad {
            position: highlight_rect.position(),
            size: highlight_rect.size(),
            color: style::highlight::HOVER(),
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
        slider_effect: false,
        };
        renderer.add_quad(&highlight_bg, Some(&content_rect));
    }
    
    // Render animated selection border (active selection, fades in/out on click)
    if app.sidebar.conversations_list.scroll_view.selection_border_visible || app.sidebar.conversations_list.scroll_view.selection_border_opacity > 0.01 {
        let opacity = app.sidebar.conversations_list.scroll_view.selection_border_opacity;
        let selection_y = app.sidebar.conversations_list.scroll_view.selection_border_y;
        let selection_rect = Rect::new(
            content_rect.x + padding,
            selection_y,
            content_rect.width - padding * 2.0,
            item_height,
        );
        let corner_radius = style::corner_radius::MEDIUM;
        let ph = style::accent::PHOSPHOR();
        let mut ring_col = Vec4::new(ph.x, ph.y, ph.z, style::stroke::SELECTION_OUTLINE_PX);
        ring_col.w *= opacity;
        
        // Render border using SDF border mode (negative corner_radius signals border)
        let border_quad = Quad {
            position: selection_rect.position(),
            size: selection_rect.size(),
            color: ring_col,
            corner_radius: -corner_radius, // Negative signals border mode
            bubble_effect: false,
        slider_effect: false,
        };
        renderer.add_quad(&border_quad, Some(&content_rect));
    }
    
    // Render conversation items (using renderer.add_vertices for batching)
    for (i, conv) in app.chat_state.conversations.iter().enumerate() {
        if let Some(item_rect) = section.item_rect(sidebar_rect, y_offset, i, padding) {
            let is_selected = app.sidebar.selected_conversation_id.as_ref() == Some(&conv.id) 
                || app.chat_state.current_conversation_id.as_ref() == Some(&conv.id);
            
            // Note: Selection border is now rendered separately using animated position
            // (see below, after the loop)
            
            // Handle button on right
            let handle_size = 24.0;
            let handle_rect = Rect::new(
                item_rect.right() - handle_size - padding,
                item_rect.y + (item_rect.height - handle_size) / 2.0,
                handle_size,
                handle_size,
            );
            
            let is_expanded = app.sidebar.conversations_list.expanded_index == Some(i);
            
            // Calculate available width for title text based on whether buttons are expanded
            // Buttons are square, with icon size matching text height (SMALL = 12.0)
            // Button is slightly larger than icon to provide padding
            let button_icon_size = ITEM_TITLE_FONT_SIZE; // Same as text height
            let button_size = button_icon_size + style::padding::TINY * 2.0; // Icon + padding on both sides
            let button_spacing = style::padding::TINY;
            
            // Get expansion animation to calculate button width
            let expand_anim = app.sidebar.conversations_list.get_expand_animation(i);
            let buttons_width = if expand_anim > 0.0 {
                // Two buttons (delete + info) + spacing + handle, interpolated by animation
                let expanded_width = (button_size + button_spacing) * 2.0 + handle_size + padding;
                let collapsed_width = handle_size + padding;
                collapsed_width + (expanded_width - collapsed_width) * expand_anim
            } else {
                // Just handle
                handle_size + padding
            };
            
            // Calculate available width for title text
            let available_title_width = item_rect.width - text_x_offset - buttons_width;
            
            // Truncate title text to fit available width
            let mut title = conv.title.clone();
            let title_width = renderer.measure_text(&title, ITEM_TITLE_FONT_SIZE).x;
            if title_width > available_title_width {
                // Binary search for the right truncation point
                let mut low = 0;
                let mut high = title.len();
                while low < high {
                    let mid = (low + high + 1) / 2;
                    let truncated = if mid < title.len() {
                        format!("{}...", &title[..mid])
                    } else {
                        title.clone()
                    };
                    let width = renderer.measure_text(&truncated, ITEM_TITLE_FONT_SIZE).x;
                    if width <= available_title_width {
                        low = mid;
                    } else {
                        high = mid - 1;
                    }
                }
                if low < title.len() {
                    title = format!("{}...", &title[..low]);
                }
            }
            
            let mut text_color = if is_selected {
                style::text::PRIMARY()
            } else {
                style::text::SECONDARY()
            };
            // Apply opacity to item text
            text_color.w *= text_opacity;
            let item_id = format!("sidebar_conversation_item_{}", i);
            // Validate item BEFORE pushing it (so it's in hierarchy when Text validates)
            renderer.validate_component(&item_id, Some("sidebar_content"), "ConversationItem");
            renderer.push_parent(item_id.clone());
            
            let title_text_rect = Rect::new(
                item_rect.x + text_x_offset,
                item_rect.y,
                available_title_width,
                item_rect.height,
            );
            let mut title_text = Text::new_for_render(&title)
                .with_font_size(ITEM_TITLE_FONT_SIZE)
                .with_color(text_color)
                .with_alignment(TextAlignment::Left)
                .with_scissor(Some(content_rect));
            title_text.update_layout(title_text_rect, None, None);
            // Create a temporary vertex buffer for text rendering
            let mut text_vertices = Vec::new();
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
            title_text.render(renderer, app, &mut text_vertices, None);
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            if !text_vertices.is_empty() {
                renderer.add_vertices(&text_vertices, Some(&content_rect));
            }
            renderer.pop_parent();
            
            // Get expansion animation value (0.0 = collapsed, 1.0 = expanded)
            let expand_anim = app.sidebar.conversations_list.get_expand_animation(i);
            
            if expand_anim > 0.0 {
                // Show action buttons with icons (animated)
                use crate::ui::icons::icon_names;
                
                // Calculate button positions with animation
                // Buttons slide in from the right (handle position)
                let button_start_x = handle_rect.x;
                let button_end_x_info = handle_rect.x - button_size - button_spacing;
                let button_end_x_delete = button_end_x_info - button_size - button_spacing;
                
                // Interpolate positions based on animation
                let info_x = button_start_x + (button_end_x_info - button_start_x) * expand_anim;
                let delete_x = button_start_x + (button_end_x_delete - button_start_x) * expand_anim;
                
                // Interpolate opacity
                let button_opacity = expand_anim;
                let mut button_icon_color = style::text::PRIMARY();
                button_icon_color.w *= button_opacity;
                
                // Info button (square)
                let info_rect = Rect::new(
                    info_x,
                    handle_rect.y + (handle_rect.height - button_size) / 2.0,
                    button_size,
                    button_size,
                );
                let mut info_bg_color = style::button::SECONDARY();
                info_bg_color.w *= button_opacity;
                let info_bg = Quad {
                    position: info_rect.position(),
                    size: info_rect.size(),
                    color: info_bg_color,
                    corner_radius: style::corner_radius::SMALL,
                    bubble_effect: false,
        slider_effect: false,
                };
                renderer.add_quad(&info_bg, Some(&content_rect));
                // Delete button (square)
                let delete_rect = Rect::new(
                    delete_x,
                    handle_rect.y + (handle_rect.height - button_size) / 2.0,
                    button_size,
                    button_size,
                );
                let mut delete_bg_color = style::button::DANGER();
                delete_bg_color.w *= button_opacity;
                let delete_bg = Quad {
                    position: delete_rect.position(),
                    size: delete_rect.size(),
                    color: delete_bg_color,
                    corner_radius: style::corner_radius::SMALL,
                    bubble_effect: false,
        slider_effect: false,
                };
                renderer.add_quad(&delete_bg, Some(&content_rect));
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                renderer.queue_icon(
                    icon_names::PENCIL,
                    Vec2::new(
                        info_rect.x + info_rect.width / 2.0 - button_icon_size / 2.0,
                        info_rect.y + info_rect.height / 2.0 - button_icon_size / 2.0,
                    ),
                    button_icon_size,
                    button_icon_color,
                );
                renderer.queue_icon(
                    icon_names::TRASH,
                    Vec2::new(
                        delete_rect.x + delete_rect.width / 2.0 - button_icon_size / 2.0,
                        delete_rect.y + delete_rect.height / 2.0 - button_icon_size / 2.0,
                    ),
                    button_icon_size,
                    button_icon_color,
                );
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            }
            // Show handle (three dots icon) - fade out when expanded
            let handle_opacity = 1.0 - expand_anim;
            let hbtn = style::button::SECONDARY();
            let mut handle_bg_color = Vec4::new(hbtn.x, hbtn.y, hbtn.z, 0.8);
            handle_bg_color.w *= handle_opacity;
            let handle_bg = Quad {
                position: handle_rect.position(),
                size: handle_rect.size(),
                color: handle_bg_color,
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
        slider_effect: false,
            };
            renderer.add_quad(&handle_bg, Some(&content_rect));
            
            if handle_opacity > 0.0 {
                // Handle icon (centered in handle button)
                use crate::ui::icons::icon_names;
                let handle_icon_size = ITEM_TITLE_FONT_SIZE; // Same size as text/other icons
                let mut handle_icon_color = style::text::SECONDARY();
                handle_icon_color.w *= handle_opacity;
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                renderer.queue_icon(
                    icon_names::DOTS_6_VERTICAL,
                    Vec2::new(
                        handle_rect.x + handle_rect.width / 2.0 - handle_icon_size / 2.0,
                        handle_rect.y + handle_rect.height / 2.0 - handle_icon_size / 2.0,
                    ),
                    handle_icon_size,
                    handle_icon_color,
                );
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            }
        }
    }
}

fn render_documents_section(
    app: &App,
    sidebar_rect: &Rect,
    y_offset: f32,
    section: &container::Section,
    renderer: &mut Renderer,
    padding: f32,
    item_height: f32,
) {
    // Calculate text animation parameters
    let min_width_for_text = 100.0;
    let text_opacity = if app.sidebar.current_width < min_width_for_text {
        (app.sidebar.current_width / min_width_for_text).max(0.0)
    } else {
        1.0
    };
    let open_width = crate::ui::sidebar::SidebarWindow::OPEN_WIDTH;
    let text_x_offset = if app.sidebar.current_width < open_width {
        (app.sidebar.current_width / open_width) * padding
    } else {
        padding
    };
    const TITLE_FONT_SIZE: f32 = style::font_size::NORMAL;
    const ITEM_TITLE_FONT_SIZE: f32 = style::font_size::SMALL;
    const FONT_SIZE: f32 = style::font_size::MEDIUM;
    
    // Render title (unclipped - title is outside scrollable area)
    let title_rect = section.title_rect(sidebar_rect, y_offset);
    
    // Use horizontal stack to position title and button
    let button_size = Vec2::new(30.0, 30.0);
    let title_text_width = renderer.measure_text("Documents", TITLE_FONT_SIZE).x;
    let title_container = Rect::new(
        title_rect.x,
        title_rect.y,
        title_rect.width,
        title_rect.height,
    );
    
    let title_rects = layout::stack_horizontal(
        &title_container,
        &[title_text_width, button_size.x],
        padding,
        padding,
    );
    
    // Documents title - use Text component
    let mut title_color = style::text::PRIMARY();
    title_color.w *= text_opacity;
    let title_text_rect = Rect::new(
        title_rects[0].x,
        title_rect.y + (title_rect.height - TITLE_FONT_SIZE * 1.2) / 2.0,
        title_rects[0].width,
        TITLE_FONT_SIZE * 1.2,
    );
    renderer.push_parent("sidebar_documents_title".to_string());
    renderer.validate_component("sidebar_documents_title", None, "SectionTitle");
    let mut title_text = Text::new_for_render("Documents")
        .with_font_size(TITLE_FONT_SIZE)
        .with_color(title_color)
        .with_alignment(TextAlignment::Left)
        .with_scissor(None);
    title_text.update_layout(title_text_rect, None, None);
    let mut text_vertices = Vec::new();
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
    title_text.render(renderer, app, &mut text_vertices, None);
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
    if !text_vertices.is_empty() {
        renderer.add_vertices(&text_vertices, None);
    }
    renderer.pop_parent();
    
    // New document button
    let button_rect = Rect::new(
        title_rect.right() - button_size.x - padding,
        title_rect.y + (title_rect.height - button_size.y) / 2.0,
        button_size.x,
        button_size.y,
    );
    let new_doc_color = match app.sidebar.new_document_button.state {
        crate::ui::ButtonState::Pressed => style::button::PRIMARY_ACTIVE(),
        crate::ui::ButtonState::Hover => style::button::PRIMARY_HOVER(),
        crate::ui::ButtonState::Normal => style::button::PRIMARY(),
    };
    let new_doc_bg = Quad {
        position: button_rect.position(),
        size: button_rect.size(),
        color: new_doc_color,
        corner_radius: style::corner_radius::MEDIUM,
        bubble_effect: false,
        slider_effect: false,
    };
    // Button is in title area - use sidebar_rect for scissor to ensure it renders
    renderer.add_quad(&new_doc_bg, Some(sidebar_rect));
    // New document button text - use Text component
    renderer.push_parent("sidebar_documents_button".to_string());
    renderer.validate_component("sidebar_documents_button", None, "Button");
    let mut plus_text = Text::new_for_render("+")
        .with_font_size(FONT_SIZE)
        .with_color(style::text::PRIMARY())
        .with_alignment(TextAlignment::Center)
        .with_scissor(None);
    plus_text.update_layout(button_rect, None, None);
    let mut text_vertices = Vec::new();
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
    plus_text.render(renderer, app, &mut text_vertices, None);
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
    if !text_vertices.is_empty() {
        renderer.add_vertices(&text_vertices, None);
    }
    renderer.pop_parent();
    
    // Get content rect for clipping scrolled content
    let content_rect = section.content_rect(sidebar_rect, y_offset);
    
    // Render unified highlight bar for documents list
    if app.sidebar.documents_list.scroll_view.highlight_bar_visible {
        let highlight_y = app.sidebar.documents_list.scroll_view.highlight_bar_y;
        let highlight_rect = Rect::new(
            content_rect.x + padding,
            highlight_y,
            content_rect.width - padding * 2.0,
            item_height,
        );
        let highlight_bg = Quad {
            position: highlight_rect.position(),
            size: highlight_rect.size(),
            color: style::highlight::HOVER(),
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
        slider_effect: false,
        };
        renderer.add_quad(&highlight_bg, Some(&content_rect));
    }
    // Selection border for documents
    if app.sidebar.documents_list.scroll_view.selection_border_visible || app.sidebar.documents_list.scroll_view.selection_border_opacity > 0.01 {
        let opacity = app.sidebar.documents_list.scroll_view.selection_border_opacity;
        let selection_y = app.sidebar.documents_list.scroll_view.selection_border_y;
        let selection_rect = Rect::new(
            content_rect.x + padding,
            selection_y,
            content_rect.width - padding * 2.0,
            item_height,
        );
        let ph = style::accent::PHOSPHOR();
        let mut ring_col = Vec4::new(ph.x, ph.y, ph.z, style::stroke::SELECTION_OUTLINE_PX);
        ring_col.w *= opacity;
        let border_quad = Quad {
            position: selection_rect.position(),
            size: selection_rect.size(),
            color: ring_col,
            corner_radius: -style::corner_radius::MEDIUM,
            bubble_effect: false,
        slider_effect: false,
        };
        renderer.add_quad(&border_quad, Some(&content_rect));
    }
    
    // Render document items with collapsible handle + buttons (same as conversations/insights)
    if let Ok(document_ids) = DocumentPersistence::list_documents() {
        let handle_size = 24.0;
        let button_icon_size = ITEM_TITLE_FONT_SIZE;
        let button_size = button_icon_size + style::padding::TINY * 2.0;
        let button_spacing = style::padding::TINY;
        for (i, doc_id) in document_ids.iter().enumerate() {
            if let Some(item_rect) = section.item_rect(sidebar_rect, y_offset, i, padding) {
                let is_selected = app.sidebar.selected_document_id.as_ref() == Some(doc_id);
                let expand_anim = app.sidebar.documents_list.get_expand_animation(i);
                let handle_rect = Rect::new(
                    item_rect.right() - handle_size - padding,
                    item_rect.y + (item_rect.height - handle_size) / 2.0,
                    handle_size,
                    handle_size,
                );
                let buttons_width = (button_size + button_spacing) * 2.0 + handle_size + padding;
                let available_title_width = item_rect.width - text_x_offset - buttons_width;
                let doc_name = if doc_id.len() > 25 {
                    format!("{}...", &doc_id[..25])
                } else {
                    doc_id.clone()
                };
                let mut text_color = if is_selected {
                    style::text::PRIMARY()
                } else {
                    style::text::SECONDARY()
                };
                text_color.w *= text_opacity;
                let doc_item_id = format!("sidebar_document_item_{}", i);
                renderer.push_parent(doc_item_id.clone());
                renderer.validate_component(&doc_item_id, None, "DocumentItem");
                let doc_name_rect = Rect::new(
                    item_rect.x + text_x_offset,
                    item_rect.y,
                    available_title_width.max(0.0),
                    item_rect.height,
                );
                let mut doc_name_text = Text::new_for_render(&doc_name)
                    .with_font_size(ITEM_TITLE_FONT_SIZE)
                    .with_color(text_color)
                    .with_alignment(TextAlignment::Left)
                    .with_scissor(Some(content_rect));
                doc_name_text.update_layout(doc_name_rect, None, None);
                let mut text_vertices = Vec::new();
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                doc_name_text.render(renderer, app, &mut text_vertices, None);
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
                if !text_vertices.is_empty() {
                    renderer.add_vertices(&text_vertices, Some(&content_rect));
                }
                renderer.pop_parent();
                if expand_anim > 0.0 {
                    use crate::ui::icons::icon_names;
                    let button_end_x_info = handle_rect.x - button_size - button_spacing;
                    let button_end_x_delete = button_end_x_info - button_size - button_spacing;
                    let info_x = handle_rect.x + (button_end_x_info - handle_rect.x) * expand_anim;
                    let delete_x = handle_rect.x + (button_end_x_delete - handle_rect.x) * expand_anim;
                    let mut btn_color = style::text::PRIMARY();
                    btn_color.w *= expand_anim;
                    let mut bg_secondary = style::button::SECONDARY();
                    bg_secondary.w *= expand_anim;
                    let mut bg_danger = style::button::DANGER();
                    bg_danger.w *= expand_anim;
                    let info_rect = Rect::new(info_x, handle_rect.y + (handle_rect.height - button_size) / 2.0, button_size, button_size);
                    let delete_rect = Rect::new(delete_x, handle_rect.y + (handle_rect.height - button_size) / 2.0, button_size, button_size);
                    renderer.add_quad(&Quad { position: info_rect.position(), size: info_rect.size(), color: bg_secondary, corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
                    renderer.add_quad(&Quad { position: delete_rect.position(), size: delete_rect.size(), color: bg_danger, corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
                    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                    renderer.queue_icon(icon_names::PENCIL, Vec2::new(info_rect.x + info_rect.width / 2.0 - button_icon_size / 2.0, info_rect.y + info_rect.height / 2.0 - button_icon_size / 2.0), button_icon_size, btn_color);
                    renderer.queue_icon(icon_names::TRASH, Vec2::new(delete_rect.x + delete_rect.width / 2.0 - button_icon_size / 2.0, delete_rect.y + delete_rect.height / 2.0 - button_icon_size / 2.0), button_icon_size, btn_color);
                    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
                }
                let handle_opacity = 1.0 - expand_anim;
                let hbtn = style::button::SECONDARY();
                let mut handle_bg_color = Vec4::new(hbtn.x, hbtn.y, hbtn.z, 0.8);
                handle_bg_color.w *= handle_opacity;
                renderer.add_quad(&Quad { position: handle_rect.position(), size: handle_rect.size(), color: handle_bg_color, corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
                if handle_opacity > 0.0 {
                    use crate::ui::icons::icon_names;
                    let mut handle_icon_color = style::text::SECONDARY();
                    handle_icon_color.w *= handle_opacity;
                    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                    renderer.queue_icon(icon_names::DOTS_6_VERTICAL, Vec2::new(handle_rect.x + handle_rect.width / 2.0 - button_icon_size / 2.0, handle_rect.y + handle_rect.height / 2.0 - button_icon_size / 2.0), button_icon_size, handle_icon_color);
                    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
                }
            }
        }
    }
}

fn render_insights_section(
    app: &App,
    sidebar_rect: &Rect,
    y_offset: f32,
    section: &container::Section,
    renderer: &mut Renderer,
    padding: f32,
) {
    // Calculate text animation parameters
    let min_width_for_text = 100.0;
    let text_opacity = if app.sidebar.current_width < min_width_for_text {
        (app.sidebar.current_width / min_width_for_text).max(0.0)
    } else {
        1.0
    };
    let open_width = crate::ui::sidebar::SidebarWindow::OPEN_WIDTH;
    let text_x_offset = if app.sidebar.current_width < open_width {
        (app.sidebar.current_width / open_width) * padding
    } else {
        padding
    };
    const TITLE_FONT_SIZE: f32 = style::font_size::NORMAL;
    const ITEM_FONT_SIZE: f32 = style::font_size::TINY;
    
    // Render title (unclipped - title is outside scrollable area)
    let title_rect = section.title_rect(sidebar_rect, y_offset);
    // Use horizontal stack layout for consistency (even though there's no button)
    let title_text_width = renderer.measure_text("Insights", TITLE_FONT_SIZE).x;
    let title_container = Rect::new(
        title_rect.x,
        title_rect.y,
        title_rect.width,
        title_rect.height,
    );
    
    let title_rects = layout::stack_horizontal(
        &title_container,
        &[title_text_width],
        padding,
        padding,
    );
    
    // Apply opacity to title text
    let mut title_color = style::text::PRIMARY();
    title_color.w *= text_opacity;
    // Insights title - use Text component
    let title_text_rect = Rect::new(
        title_rects[0].x,
        title_rect.y + (title_rect.height - TITLE_FONT_SIZE * 1.2) / 2.0,
        title_rects[0].width,
        TITLE_FONT_SIZE * 1.2,
    );
    renderer.push_parent("sidebar_insights_title".to_string());
    renderer.validate_component("sidebar_insights_title", None, "SectionTitle");
    let mut title_text = Text::new_for_render("Insights")
        .with_font_size(TITLE_FONT_SIZE)
        .with_color(title_color)
        .with_alignment(TextAlignment::Left)
        .with_scissor(None);
    title_text.update_layout(title_text_rect, None, None);
    let mut text_vertices = Vec::new();
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
    title_text.render(renderer, app, &mut text_vertices, None);
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
    if !text_vertices.is_empty() {
        renderer.add_vertices(&text_vertices, None);
    }
    renderer.pop_parent();
    
    // Get content rect for clipping scrolled content
    let content_rect = section.content_rect(sidebar_rect, y_offset);
    const INSIGHTS_ITEM_HEIGHT: f32 = 35.0;

    // Highlight bar (hover)
    if app.sidebar.insights_panel.insights_list.scroll_view.highlight_bar_visible {
        let highlight_y = app.sidebar.insights_panel.insights_list.scroll_view.highlight_bar_y;
        let highlight_rect = Rect::new(
            content_rect.x + padding,
            highlight_y,
            content_rect.width - padding * 2.0,
            INSIGHTS_ITEM_HEIGHT,
        );
        let highlight_bg = Quad {
            position: highlight_rect.position(),
            size: highlight_rect.size(),
            color: style::highlight::HOVER(),
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
        slider_effect: false,
        };
        renderer.add_quad(&highlight_bg, Some(&content_rect));
    }
    // Selection border
    if app.sidebar.insights_panel.insights_list.scroll_view.selection_border_visible || app.sidebar.insights_panel.insights_list.scroll_view.selection_border_opacity > 0.01 {
        let opacity = app.sidebar.insights_panel.insights_list.scroll_view.selection_border_opacity;
        let selection_y = app.sidebar.insights_panel.insights_list.scroll_view.selection_border_y;
        let selection_rect = Rect::new(
            content_rect.x + padding,
            selection_y,
            content_rect.width - padding * 2.0,
            INSIGHTS_ITEM_HEIGHT,
        );
        let ph = style::accent::PHOSPHOR();
        let mut ring_col = Vec4::new(ph.x, ph.y, ph.z, style::stroke::SELECTION_OUTLINE_PX);
        ring_col.w *= opacity;
        let border_quad = Quad {
            position: selection_rect.position(),
            size: selection_rect.size(),
            color: ring_col,
            corner_radius: -style::corner_radius::MEDIUM,
            bubble_effect: false,
        slider_effect: false,
        };
        renderer.add_quad(&border_quad, Some(&content_rect));
    }

    // Icon sizes for insight row buttons (edit, delete, drag handle)
    let button_icon_size = style::font_size::TINY;
    let button_size = button_icon_size + style::padding::TINY * 2.0;
    let button_spacing = style::padding::TINY;
    let handle_size = 24.0;
    let buttons_width = (button_size + button_spacing) * 2.0 + handle_size + padding;

    // Render insight items (text + collapsible edit/delete/drag like conversations)
    for (i, insight) in app.insights_state.insights.iter().enumerate() {
        if let Some(item_rect) = section.item_rect(sidebar_rect, y_offset, i, padding) {
            let is_selected = app.sidebar.selected_insight_id.as_ref() == Some(&insight.id);
            let is_hovered = app.sidebar.hovered_insight_id.as_ref() == Some(&insight.id);
            let expand_anim = app.sidebar.insights_panel.insights_list.get_expand_animation(i);
            
            // Item background (hover/selected handled by highlight bar and selection border)
            let item_color = if is_selected {
                style::highlight::HOVER()
            } else if is_hovered {
                let a = style::highlight::ACTIVE();
                Vec4::new(a.x, a.y, a.z, 0.9)
            } else {
                let w = style::bg::PANEL_WELL();
                Vec4::new(w.x, w.y, w.z, 0.55)
            };
            let item_bg = Quad {
                position: item_rect.position(),
                size: item_rect.size(),
                color: item_color,
                corner_radius: style::corner_radius::MEDIUM,
                bubble_effect: false,
        slider_effect: false,
            };
            renderer.add_quad(&item_bg, Some(&content_rect));
            
            let available_text_width = item_rect.width - text_x_offset - buttons_width;
            let display_text = if !insight.title.is_empty() {
                if insight.title.len() > 30 {
                    format!("{}...", &insight.title[..30])
                } else {
                    insight.title.clone()
                }
            } else if insight.text.len() > 30 {
                format!("{}...", &insight.text[..30])
            } else {
                insight.text.clone()
            };
            let mut text_color = if is_selected {
                style::text::PRIMARY()
            } else {
                style::text::SECONDARY()
            };
            text_color.w *= text_opacity;
            let insight_item_id = format!("sidebar_insight_item_{}", i);
            renderer.push_parent(insight_item_id.clone());
            renderer.validate_component(&insight_item_id, None, "InsightItem");
            let insight_text_rect = Rect::new(
                item_rect.x + text_x_offset,
                item_rect.y,
                available_text_width.max(0.0),
                item_rect.height,
            );
            let mut insight_text = Text::new_for_render(&display_text)
                .with_font_size(ITEM_FONT_SIZE)
                .with_color(text_color)
                .with_alignment(TextAlignment::Left)
                .with_scissor(Some(content_rect));
            insight_text.update_layout(insight_text_rect, None, None);
            let mut text_vertices = Vec::new();
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
            insight_text.render(renderer, app, &mut text_vertices, None);
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            if !text_vertices.is_empty() {
                renderer.add_vertices(&text_vertices, Some(&content_rect));
            }
            renderer.pop_parent();

            use crate::ui::icons::icon_names;
            let handle_rect = Rect::new(
                item_rect.right() - padding - handle_size,
                item_rect.y + (item_rect.height - handle_size) / 2.0,
                handle_size,
                handle_size,
            );
            let trash_rect = Rect::new(
                handle_rect.x - button_spacing - button_size,
                item_rect.y + (item_rect.height - button_size) / 2.0,
                button_size,
                button_size,
            );
            let pencil_rect = Rect::new(
                trash_rect.x - button_spacing - button_size,
                item_rect.y + (item_rect.height - button_size) / 2.0,
                button_size,
                button_size,
            );
            let mut icon_color = style::text::SECONDARY();
            icon_color.w *= text_opacity;
            if expand_anim > 0.0 {
                let button_start_x = handle_rect.x;
                let button_end_x_pencil = pencil_rect.x;
                let button_end_x_trash = trash_rect.x;
                let pencil_x = button_start_x + (button_end_x_pencil - button_start_x) * expand_anim;
                let trash_x = button_start_x + (button_end_x_trash - button_start_x) * expand_anim;
                let mut btn_color = style::text::PRIMARY();
                btn_color.w *= expand_anim;
                let mut bg_secondary = style::button::SECONDARY();
                bg_secondary.w *= expand_anim;
                let mut bg_danger = style::button::DANGER();
                bg_danger.w *= expand_anim;
                let anim_pencil_rect = Rect::new(pencil_x, pencil_rect.y, button_size, button_size);
                let anim_trash_rect = Rect::new(trash_x, trash_rect.y, button_size, button_size);
                renderer.add_quad(&Quad { position: anim_pencil_rect.position(), size: anim_pencil_rect.size(), color: bg_secondary, corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
                renderer.add_quad(&Quad { position: anim_trash_rect.position(), size: anim_trash_rect.size(), color: bg_danger, corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                renderer.queue_icon(icon_names::PENCIL, Vec2::new(anim_pencil_rect.x + anim_pencil_rect.width / 2.0 - button_icon_size / 2.0, anim_pencil_rect.y + anim_pencil_rect.height / 2.0 - button_icon_size / 2.0), button_icon_size, btn_color);
                renderer.queue_icon(icon_names::TRASH, Vec2::new(anim_trash_rect.x + anim_trash_rect.width / 2.0 - button_icon_size / 2.0, anim_trash_rect.y + anim_trash_rect.height / 2.0 - button_icon_size / 2.0), button_icon_size, btn_color);
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            }
            let handle_opacity = 1.0 - expand_anim;
            let hbtn = style::button::SECONDARY();
            let mut handle_bg_color = Vec4::new(hbtn.x, hbtn.y, hbtn.z, 0.8);
            handle_bg_color.w *= handle_opacity;
            renderer.add_quad(&Quad { position: handle_rect.position(), size: handle_rect.size(), color: handle_bg_color, corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
            if handle_opacity > 0.0 {
                icon_color.w *= handle_opacity;
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                renderer.queue_icon(icon_names::DOTS_6_VERTICAL, Vec2::new(handle_rect.x + handle_rect.width / 2.0 - button_icon_size / 2.0, handle_rect.y + handle_rect.height / 2.0 - button_icon_size / 2.0), button_icon_size, icon_color);
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            }
        }
    }
}

fn render_collections_section(
    app: &App,
    sidebar_rect: &Rect,
    y_offset: f32,
    section: &container::Section,
    renderer: &mut Renderer,
    padding: f32,
    item_height: f32,
) {
    use crate::ui::icons::icon_names;
    let Some(library) = app.library_window.as_ref() else { return; };
    let title_rect = section.title_rect(sidebar_rect, y_offset);
    let title_text_rect = Rect::new(title_rect.x + padding, title_rect.y, title_rect.width - padding * 2.0, title_rect.height);
    renderer.push_parent("sidebar_collections_title".to_string());
    renderer.validate_component("sidebar_collections_title", None, "SectionTitle");
    let mut title_text = Text::new_for_render("Collections")
        .with_font_size(style::font_size::NORMAL)
        .with_color(style::text::PRIMARY())
        .with_alignment(TextAlignment::Left);
    title_text.update_layout(title_text_rect, None, None);
    let mut title_vertices = Vec::new();
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
    title_text.render(renderer, app, &mut title_vertices, None);
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
    renderer.add_vertices(&title_vertices, None);
    renderer.pop_parent();

    let content_rect = section.content_rect(sidebar_rect, y_offset);
    if app.sidebar.collections_list.scroll_view.highlight_bar_visible {
        let y = app.sidebar.collections_list.scroll_view.highlight_bar_y;
        renderer.add_quad(&Quad {
            position: Vec2::new(content_rect.x + padding, y),
            size: Vec2::new(content_rect.width - padding * 2.0, item_height),
            color: style::highlight::HOVER(),
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
            slider_effect: false,
        }, Some(&content_rect));
    }
    for (i, collection) in library.collections.iter().enumerate() {
        if let Some(item_rect) = section.item_rect(sidebar_rect, y_offset, i, padding) {
            let handle_size = 24.0;
            let handle_rect = Rect::new(item_rect.right() - handle_size - padding, item_rect.y + (item_rect.height - handle_size) * 0.5, handle_size, handle_size);
            let is_expanded = app.sidebar.collections_list.expanded_index == Some(i);
            let button_size = style::font_size::SMALL + style::padding::TINY * 2.0;
            let button_spacing = style::padding::TINY;
            let expand_anim = app.sidebar.collections_list.get_expand_animation(i);
            let available = item_rect.width - padding - handle_size - (button_size * 2.0 + button_spacing * 2.0) * expand_anim;
            let mut text = format!("{} ({})", collection.name, collection.paper_count);
            if renderer.measure_text(&text, style::font_size::SMALL).x > available {
                text.truncate(24);
                text.push_str("...");
            }
            let mut row_text = Text::new_for_render(&text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(TextAlignment::Left)
                .with_scissor(Some(content_rect));
            row_text.update_layout(Rect::new(item_rect.x + padding, item_rect.y, available.max(0.0), item_rect.height), None, None);
            let mut text_vertices = Vec::new();
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
            row_text.render(renderer, app, &mut text_vertices, None);
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            renderer.add_vertices(&text_vertices, Some(&content_rect));

            if is_expanded || expand_anim > 0.0 {
                let info_rect = Rect::new(handle_rect.x - button_size - button_spacing, handle_rect.y + (handle_rect.height - button_size) * 0.5, button_size, button_size);
                let del_rect = Rect::new(info_rect.x - button_size - button_spacing, info_rect.y, button_size, button_size);
                renderer.add_quad(&Quad { position: info_rect.position(), size: info_rect.size(), color: style::button::SECONDARY(), corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
                renderer.add_quad(&Quad { position: del_rect.position(), size: del_rect.size(), color: style::button::DANGER(), corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
                let icon_pad = (button_size - 12.0) * 0.5;
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
                renderer.queue_icon(icon_names::PENCIL, Vec2::new(info_rect.x + icon_pad, info_rect.y + icon_pad), 12.0, style::text::PRIMARY());
                renderer.queue_icon(icon_names::TRASH, Vec2::new(del_rect.x + icon_pad, del_rect.y + icon_pad), 12.0, style::text::PRIMARY());
                renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
            }
            renderer.add_quad(&Quad { position: handle_rect.position(), size: handle_rect.size(), color: style::button::SECONDARY(), corner_radius: style::corner_radius::SMALL, bubble_effect: false, slider_effect: false }, Some(&content_rect));
            let icon_pad = (handle_rect.size().x - 12.0) * 0.5;
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
            renderer.queue_icon(icon_names::DOTS_6_VERTICAL, Vec2::new(handle_rect.x + icon_pad, handle_rect.y + icon_pad), 12.0, style::text::PRIMARY());
            renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
        }
    }
}

fn render_settings_button(app: &App, _sidebar_rect: &Rect, renderer: &mut Renderer) {
    use crate::ui::icons::icon_names;

    let btn = &app.sidebar.settings_button;
    let btn_rect = Rect::new(btn.position.x, btn.position.y, btn.size.x, btn.size.y);

    let bg_color = match btn.state {
        crate::ui::ButtonState::Pressed => style::button::PRIMARY_ACTIVE(),
        crate::ui::ButtonState::Hover => style::button::PRIMARY_HOVER(),
        crate::ui::ButtonState::Normal => style::button::SECONDARY(),
    };

    renderer.add_quad(
        &Quad {
            position: btn_rect.position(),
            size: btn_rect.size(),
            color: bg_color,
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
            slider_effect: false,
        },
        None,
    );

    let icon_size = 16.0;
    let icon_x = btn_rect.x + (btn_rect.width - icon_size) * 0.5;
    let icon_y = btn_rect.y + (btn_rect.height - icon_size) * 0.5;
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
    renderer.queue_icon(icon_names::GEAR, Vec2::new(icon_x, icon_y), icon_size, style::text::PRIMARY());
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
}

const SETTINGS_ACTIONS: &[(&str, &str)] = &[
    (crate::ui::icons::icon_names::GEAR,   "Reset graph layout"),
    (crate::ui::icons::icon_names::MAGNIFY, "Fit to view"),
    (crate::ui::icons::icon_names::MINIMIZE, "Reset zoom to 100%"),
    (crate::ui::icons::icon_names::CLOSE,  "Clear muted messages"),
    (crate::ui::icons::icon_names::CLOSE,  "Collapse all citations"),
];

fn render_settings_panel(app: &App, _sidebar_rect: &Rect, renderer: &mut Renderer) {
    const ITEM_HEIGHT: f32 = 36.0;
    const PANEL_PADDING: f32 = 6.0;
    const ICON_SIZE: f32 = 14.0;
    const ICON_MARGIN: f32 = 10.0;
    const FONT_SIZE: f32 = style::font_size::SMALL;

    let (panel_pos, panel_size) = app.sidebar.settings_panel_rect();
    let panel_rect = Rect::new(panel_pos.x, panel_pos.y, panel_size.x, panel_size.y);

    // Panel background
    renderer.add_quad(
        &Quad {
            position: panel_rect.position(),
            size: panel_rect.size(),
            color: style::bg::PANEL_POPUP(),
            corner_radius: style::corner_radius::LARGE,
            bubble_effect: false,
            slider_effect: false,
        },
        None,
    );

    for (i, &(icon_name, label)) in SETTINGS_ACTIONS.iter().enumerate() {
        let item_y = panel_pos.y + PANEL_PADDING + i as f32 * ITEM_HEIGHT;
        let item_rect = Rect::new(panel_pos.x + PANEL_PADDING, item_y, panel_size.x - PANEL_PADDING * 2.0, ITEM_HEIGHT);

        // Hover highlight
        if app.sidebar.settings_panel_hovered_item == Some(i) {
            renderer.add_quad(
                &Quad {
                    position: item_rect.position(),
                    size: item_rect.size(),
                    color: style::highlight::HOVER(),
                    corner_radius: style::corner_radius::MEDIUM,
                    bubble_effect: false,
                    slider_effect: false,
                },
                None,
            );
        }

        let icon_x = item_rect.x + ICON_MARGIN;
        let icon_y = item_rect.y + (ITEM_HEIGHT - ICON_SIZE) * 0.5;
        renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::HudChrome);
        renderer.queue_icon(icon_name, Vec2::new(icon_x, icon_y), ICON_SIZE, style::text::SECONDARY());

        let text_x = icon_x + ICON_SIZE + ICON_MARGIN;
        let text_rect = Rect::new(
            text_x,
            item_rect.y + (ITEM_HEIGHT - FONT_SIZE * 1.4) * 0.5,
            item_rect.right() - text_x - ICON_MARGIN,
            FONT_SIZE * 1.4,
        );
        let component_id = format!("sidebar_settings_action_{}", i);
        renderer.validate_component(&component_id, Some("sidebar_content"), "SettingsActionLabel");
        renderer.push_parent(component_id.clone());
        let mut label_text = Text::new_for_render(label)
            .with_font_size(FONT_SIZE)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Left)
            .with_scissor(None);
        label_text.update_layout(text_rect, None, None);
        let mut text_verts = Vec::new();
        label_text.render(renderer, app, &mut text_verts, None);
        if !text_verts.is_empty() {
            renderer.add_vertices(&text_verts, None);
        }
        renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::SidebarChrome);
        renderer.pop_parent();
    }
}

/// Stateless [`Renderable`] for sidebar document list; delegates to [`render_sidebar_content`].
pub struct SidebarContentViewport;

pub const SIDEBAR_CONTENT_VIEWPORT: SidebarContentViewport = SidebarContentViewport;

/// Opt-in drop shadow for the sidebar content region.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for SidebarContentViewport {
    fn z_order(&self) -> i32 {
        20
    }

    fn render(
        &self,
        renderer: &mut Renderer,
        app: &App,
        vertices: &mut Vec<Vertex>,
        _dirty_rect: Option<Rect>,
    ) {
        if let Some(spec) = SHADOW.get() {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), &spec);
            }
        }
        render_sidebar_content(renderer, app, vertices);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        let y = app.header.size.y;
        let h = app.viewport_size.y - y;
        let open_width = crate::ui::sidebar::SidebarWindow::OPEN_WIDTH;
        let width_delta = open_width - app.sidebar.current_width;
        let translation_offset = -width_delta;
        Some(Rect::new(
            app.sidebar.position.x + translation_offset,
            y,
            app.sidebar.current_width,
            h,
        ))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}

