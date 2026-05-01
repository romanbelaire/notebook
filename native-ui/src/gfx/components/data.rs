use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::core::Rect;
use crate::ui::{Text, TextAlignment};
use crate::ui::ButtonState;
use crate::ui::components::{Renderable, VStack};
use crate::ui::ingest_window::{
    INGEST_BUTTON_ROW_GAP, INGEST_INPUT_HEIGHT, INGEST_LABEL_HEIGHT, INGEST_PADDING,
    INGEST_SECTION_SPACING, INGEST_TITLE_HEIGHT, INGEST_VSTACK_SPACING,
};

const DATA_TAB_BUTTON_HOVER_OUTSET: f32 = 1.5;

fn data_tab_button_fill(state: ButtonState, disabled: bool) -> Vec4 {
    if disabled {
        return style::button::PRIMARY() * Vec4::new(0.65, 0.65, 0.65, 1.0);
    }
    match state {
        ButtonState::Pressed => style::button::PRIMARY_ACTIVE(),
        ButtonState::Hover => style::button::PRIMARY_HOVER(),
        ButtonState::Normal => style::button::PRIMARY(),
    }
}

/// Fill + optional hover outline ring (outer border quad, inner fill).
fn push_data_tab_button_background(
    vertices: &mut Vec<Vertex>,
    rect: Rect,
    state: ButtonState,
    disabled: bool,
) {
    let fill = data_tab_button_fill(state, disabled);
    if !disabled && state == ButtonState::Hover {
        let w = DATA_TAB_BUTTON_HOVER_OUTSET;
        let outer = Quad {
            position: Vec2::new(rect.x - w, rect.y - w),
            size: Vec2::new(rect.width + w * 2.0, rect.height + w * 2.0),
            color: style::border::HOVER(),
            corner_radius: style::corner_radius::SMALL + 1.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&outer.to_vertices());
    }
    let inner = Quad {
        position: rect.position(),
        size: rect.size(),
        color: fill,
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&inner.to_vertices());
}

pub fn render_data(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if app.ui_state.active_tab != Tab::Data {
        return;
    }
    
    // Set parent context for data components
    // Note: "data" is already validated by the renderer as a RenderableComponent
    // We just need to push it as parent for child components
    renderer.push_parent("data".to_string());
    
    if let Some(ref ingest) = app.ingest_window {
        let bg = Quad {
            position: ingest.position,
            size: ingest.size,
            color: style::bg::PRIMARY(),
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        renderer.add_quad(&bg, None);
        renderer.set_composite_layer(CompositeLayer::HudChrome);

        const PADDING: f32 = INGEST_PADDING;
        const SECTION_SPACING: f32 = INGEST_SECTION_SPACING;
        
        // Create container for vertical stacking
        let container = Rect::new(
            ingest.position.x + PADDING,
            ingest.position.y + PADDING,
            ingest.size.x - PADDING * 2.0,
            ingest.size.y - PADDING * 2.0,
        );
        
        // Title - use Text component (standalone)
        renderer.push_parent("data_title".to_string());
        renderer.validate_component("data_title", Some("data"), "Title");
        let title_rect = Rect::new(container.x, container.y, container.width, INGEST_TITLE_HEIGHT);
        let mut title_text = Text::new_for_render("Data Ingestion")
            .with_font_size(style::font_size::LARGE)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Left);
        title_text.update_layout(title_rect, None, None);
        title_text.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // ArXiv label + input row + Submit (layout matches IngestWindow::update_layout)
        let input_section_y = container.y + INGEST_TITLE_HEIGHT + SECTION_SPACING;
        let label_rect = Rect::new(
            container.x,
            input_section_y,
            container.width,
            INGEST_LABEL_HEIGHT,
        );
        renderer.validate_component("data_input_section", Some("data"), "DataInputSection");
        renderer.push_parent("data_input_section".to_string());

        let mut label_text = Text::new_for_render("ArXiv IDs / URLs:")
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::SECONDARY())
            .with_alignment(TextAlignment::Left);
        label_text.update_layout(label_rect, None, None);
        label_text.render(renderer, app, vertices, None);

        let mut pdf_input = ingest.pdf_dir_input.clone();
        pdf_input.cursor_visible = app.cursor_visible;
        pdf_input.cursor_animation_value = app.cursor_position_animation.value;
        pdf_input.render(renderer, app, vertices, None);

        let submit_rect = Rect::from_pos_size(ingest.submit_button.position, ingest.submit_button.size);
        push_data_tab_button_background(
            vertices,
            submit_rect,
            ingest.submit_button.state,
            ingest.is_ingesting,
        );
        let mut submit_label = crate::ui::text::Text::new_for_render(&ingest.submit_button.label)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Center);
        submit_label.update_layout(submit_rect, None, None);
        renderer.push_parent("data_submit_button".to_string());
        renderer.validate_component("data_submit_button", Some("data"), "SubmitButton");
        submit_label.render(renderer, app, vertices, None);
        renderer.pop_parent();

        renderer.pop_parent();

        let input_stack_height =
            INGEST_LABEL_HEIGHT + INGEST_VSTACK_SPACING + INGEST_INPUT_HEIGHT;
        let button_y = input_section_y + input_stack_height + INGEST_BUTTON_ROW_GAP;

        // Upload .bib button - position after input section
        let bib_rect = crate::ui::core::Rect::new(
            container.x,
            button_y,
            ingest.bib_upload_button.size.x,
            ingest.bib_upload_button.size.y,
        );
        push_data_tab_button_background(
            vertices,
            bib_rect,
            ingest.bib_upload_button.state,
            false,
        );

        let mut bib_text = crate::ui::text::Text::new_for_render(&ingest.bib_upload_button.label)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Center);
        bib_text.update_layout(bib_rect, None, None);

        renderer.push_parent("data_bib_button".to_string());
        renderer.validate_component("data_bib_button", Some("data"), "BibUploadButton");
        bib_text.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // Browse button - position after bib button
        let browse_rect = crate::ui::core::Rect::new(
            container.x + ingest.bib_upload_button.size.x + 10.0,
            button_y,
            ingest.browse_button.size.x,
            ingest.browse_button.size.y,
        );
        push_data_tab_button_background(
            vertices,
            browse_rect,
            ingest.browse_button.state,
            false,
        );
        
        let _browse_text_width = renderer.measure_text(&ingest.browse_button.label, style::font_size::NORMAL).x;
        // Render browse button label using Text component
        let browse_text_rect = Rect::new(
            browse_rect.x,
            browse_rect.y,
            browse_rect.width,
            browse_rect.height,
        );
        
        let mut browse_text = crate::ui::text::Text::new_for_render(&ingest.browse_button.label)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Center);
        browse_text.update_layout(browse_text_rect, None, None);
        
        renderer.push_parent("data_browse_button".to_string());
        renderer.validate_component("data_browse_button", Some("data"), "BrowseButton");
        browse_text.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // Ingest button - position after browse button
        let ingest_rect = crate::ui::core::Rect::new(
            container.x + ingest.bib_upload_button.size.x + ingest.browse_button.size.x + 20.0,
            button_y,
            ingest.ingest_button.size.x,
            ingest.ingest_button.size.y,
        );
        push_data_tab_button_background(
            vertices,
            ingest_rect,
            ingest.ingest_button.state,
            ingest.is_ingesting,
        );
        
        // Render ingest button label using Text component
        let ingest_text = if ingest.is_ingesting { "Ingesting..." } else { &ingest.ingest_button.label };
        let ingest_text_rect = Rect::new(
            ingest_rect.x,
            ingest_rect.y,
            ingest_rect.width,
            ingest_rect.height,
        );
        
        let mut ingest_text_component = crate::ui::text::Text::new_for_render(ingest_text)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Center);
        ingest_text_component.update_layout(ingest_text_rect, None, None);
        
        renderer.push_parent("data_ingest_button".to_string());
        renderer.validate_component("data_ingest_button", Some("data"), "IngestButton");
        ingest_text_component.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // Status section — register `data_status_section` in the hierarchy before VStack
        // (VStack resolves parent from the stack; parent id must exist in component_hierarchy)
        let status_section_y = button_y + ingest.ingest_button.size.y + SECTION_SPACING;
        let mut status_block_height = 0.0_f32;
        if !ingest.status_text.is_empty() || !ingest.import_summary_line.is_empty() {
            let has_summary = !ingest.import_summary_line.is_empty();
            status_block_height = if has_summary { 100.0 } else { 60.0 };
            let status_section_rect = Rect::new(
                container.x,
                status_section_y,
                container.width,
                status_block_height,
            );

            let mut status_stack = VStack::new(10.0, 0.0);
            status_stack.add_text_styled(
                "Status:",
                style::font_size::NORMAL,
                style::text::SECONDARY(),
                TextAlignment::Left,
            );
            if !ingest.status_text.is_empty() {
                status_stack.add_text_styled(
                    &ingest.status_text,
                    style::font_size::NORMAL,
                    style::text::PRIMARY(),
                    TextAlignment::Left,
                );
            }
            if has_summary {
                status_stack.add_text_styled(
                    &ingest.import_summary_line,
                    style::font_size::NORMAL,
                    style::text::SECONDARY(),
                    TextAlignment::Left,
                );
            }

            renderer.push_parent("data_status_section".to_string());
            renderer.validate_component("data_status_section", Some("data"), "DataStatusSection");
            status_stack.update_layout(status_section_rect, None, None);
            status_stack.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }

        // View failures (after last ingest with failures) — position from IngestWindow (hit-test aligned)
        if ingest.show_view_failures_button {
            let vf_rect = Rect::new(
                ingest.view_failures_button.position.x,
                ingest.view_failures_button.position.y,
                ingest.view_failures_button.size.x,
                ingest.view_failures_button.size.y,
            );
            push_data_tab_button_background(
                vertices,
                vf_rect,
                ingest.view_failures_button.state,
                false,
            );
            let mut vf_label = crate::ui::text::Text::new_for_render(&ingest.view_failures_button.label)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY())
                .with_alignment(crate::ui::text::TextAlignment::Center);
            vf_label.update_layout(vf_rect, None, None);
            renderer.push_parent("data_view_failures_button".to_string());
            renderer.validate_component("data_view_failures_button", Some("data"), "ViewFailuresButton");
            vf_label.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }

        let extra_below_status = if ingest.show_view_failures_button {
            status_block_height + ingest.view_failures_button.size.y + 20.0
        } else {
            status_block_height
        };

        // Progress bar (if ingesting)
        if ingest.is_ingesting && ingest.progress > 0.0 {
            let progress_bar_y = if !ingest.status_text.is_empty() || !ingest.import_summary_line.is_empty() {
                status_section_y + extra_below_status + SECTION_SPACING
            } else {
                status_section_y
            };
            let progress_bar_rect = Rect::new(
                container.x,
                progress_bar_y,
                container.width,
                20.0,
            );

            // Progress bar background
            let progress_bg = Quad {
                position: progress_bar_rect.position(),
                size: progress_bar_rect.size(),
                color: style::bg::SECONDARY(),
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&progress_bg.to_vertices());

            // Progress bar fill
            let progress_fill_width = progress_bar_rect.width * ingest.progress.min(1.0);
            let progress_fill = Quad {
                position: progress_bar_rect.position(),
                size: Vec2::new(progress_fill_width, progress_bar_rect.height),
                color: style::button::PRIMARY(),
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&progress_fill.to_vertices());

            // Progress percentage text - use Text component
            renderer.push_parent("data_progress_text".to_string());
            renderer.validate_component("data_progress_text", Some("data"), "ProgressText");
            let progress_text = format!("{:.0}%", ingest.progress * 100.0);
            let mut progress_text_component = Text::new_for_render(&progress_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::PRIMARY())
                .with_alignment(TextAlignment::Center);
            progress_text_component.update_layout(progress_bar_rect, None, None);
            progress_text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }

        // Drag & drop zone hint (positioned at bottom)
        let drop_zone_rect = Rect::new(
            ingest.position.x + PADDING,
            ingest.position.y + ingest.size.y - 100.0,
            ingest.size.x - PADDING * 2.0,
            80.0,
        );
        
        let drop_zone_bg = Quad {
            position: drop_zone_rect.position(),
            size: drop_zone_rect.size(),
            color: style::bg::SECONDARY(),
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&drop_zone_bg.to_vertices());
        
        // Drop hint - use Text component
        renderer.push_parent("data_drop_hint".to_string());
        renderer.validate_component("data_drop_hint", Some("data"), "DropHint");
        let mut drop_hint_text = Text::new_for_render("Drag & drop PDF files here to ingest")
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::SECONDARY())
            .with_alignment(TextAlignment::Center);
        drop_hint_text.update_layout(drop_zone_rect, None, None);
        drop_hint_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }
    
    // Pop data parent
    renderer.pop_parent();
}

/// Stateless [`Renderable`] for the Data tab; delegates to [`render_data`].
pub struct DataViewport;

pub const DATA_VIEWPORT: DataViewport = DataViewport;

/// Opt-in drop shadow for the data tab chassis.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for DataViewport {
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
        render_data(renderer, app, vertices);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.ingest_window
            .as_ref()
            .map(|w| Rect::new(w.position.x, w.position.y, w.size.x, w.size.y))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}

