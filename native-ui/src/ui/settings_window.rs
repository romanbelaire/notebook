use glam::Vec2;
use crate::ui::core::Rect;
use crate::ui::{ScrollView, TextInput, VStack, Button, Dropdown, DropdownItem};

pub struct SettingsWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub scroll_view: ScrollView,
    pub hf_token_input: TextInput,
    pub model_id_input: TextInput,
    pub openai_model_input: TextInput,
    pub theme_dropdown: Dropdown,
    /// Clickable rect for the provider row (toggle Local / OpenAI)
    pub provider_selector_rect: Rect,
    // Persistent VStacks for each section (avoid recreating each frame)
    pub model_settings_stack: VStack,
    pub generation_settings_stack: VStack,
    pub manage_system_prompts_button: Button,
}

impl SettingsWindow {
    pub fn new(position: Vec2, size: Vec2, theme_id: &str) -> Self {
        let padding = 20.0;
        let input_height = 40.0;
        
        let scroll_view = ScrollView::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(size.x - padding * 2.0, size.y - padding * 2.0),
        );

        let hf_token_input = TextInput::new(
            Vec2::new(position.x + padding, position.y + padding + 60.0),
            Vec2::new(size.x - padding * 2.0, input_height),
        );

        let model_id_input = TextInput::new(
            Vec2::new(position.x + padding, position.y + padding + 120.0),
            Vec2::new(size.x - padding * 2.0, input_height),
        );

        let openai_model_input = TextInput::new(
            Vec2::new(position.x + padding, position.y + padding + 120.0),
            Vec2::new(size.x - padding * 2.0, input_height),
        );

        use crate::ui::{TextAlignment, style};
        
        // Initialize persistent VStacks (model_settings_stack children rebuilt in render from provider)
        let mut model_settings_stack = VStack::new(10.0, 0.0);
        model_settings_stack.add_text_styled("Provider:", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
        model_settings_stack.add_text_styled("Local model", style::font_size::NORMAL, style::text::PRIMARY(), TextAlignment::Left);
        model_settings_stack.add_text_styled("Model ID:", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
        model_settings_stack.add_text_styled("HuggingFace API Key:", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
        
        let mut generation_settings_stack = VStack::new(10.0, 0.0);
        generation_settings_stack.add_text_styled("Named system prompts (/name in chat):", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
        generation_settings_stack.add_text_styled("Use Manage to add or remove prompts.", style::font_size::SMALL, style::text::SECONDARY(), TextAlignment::Left);
        
        let mut theme_dropdown = Dropdown::new(Vec2::ZERO, Vec2::new(280.0, 36.0));
        theme_dropdown.show_create_footer = false;
        for (i, (_slug, label)) in crate::ui::theme::THEME_CHOICES.iter().enumerate() {
            theme_dropdown.items.push(DropdownItem {
                id: Some(i as i32),
                label: (*label).to_string(),
                slash_name: None,
            });
        }
        let idx = crate::ui::theme::theme_index_for_id(theme_id);
        theme_dropdown.selected_index = Some(idx);
        
        Self {
            position,
            size,
            scroll_view,
            hf_token_input,
            model_id_input,
            openai_model_input,
            theme_dropdown,
            provider_selector_rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            model_settings_stack,
            generation_settings_stack,
            manage_system_prompts_button: Button::new(Vec2::ZERO, Vec2::new(240.0, 36.0), "Manage system prompts"),
        }
    }

    /// Update layout; pass current provider so local vs openai layout is correct.
    pub fn update_layout(&mut self, provider: &str) {
        use crate::ui::core::{layout, container::{SectionStack, Section}};
        
        let padding = 20.0;
        const SECTION_SPACING: f32 = 30.0;
        const TITLE_OFFSET: f32 = 50.0;

        self.scroll_view.position = Vec2::new(
            self.position.x + padding,
            self.position.y + padding,
        );
        self.scroll_view.size = Vec2::new(
            self.size.x - padding * 2.0,
            self.size.y - padding * 2.0,
        );

        let container_rect = Rect::new(
            self.position.x + padding,
            self.position.y + padding,
            self.size.x - padding * 2.0,
            self.size.y - padding * 2.0,
        );

        let mut section_stack = SectionStack::new(SECTION_SPACING);
        let mut model_section = Section::new("Model Settings".to_string(), 40.0);
        let (item_heights, model_input_idx, hf_idx, openai_idx) = if provider == "openai" {
            model_section.item_count = 5; // Provider label, value, OpenAI model label, input, note
            (vec![25.0, 25.0, 25.0, 40.0, 25.0], 3, None, Some(3))
        } else {
            model_section.item_count = 6; // Provider label, value, Model ID label, input, HF label, input
            (vec![25.0, 25.0, 25.0, 40.0, 25.0, 40.0], 3, Some(5), None)
        };
        model_section.title_height = 40.0;
        section_stack.add_section(model_section);

        let layout = section_stack.layout(&container_rect);
        let section_0_y_offset = layout.iter().find(|(idx, _)| *idx == 0)
            .map(|(_, y)| *y)
            .unwrap_or(0.0);
        let section_y = container_rect.y + TITLE_OFFSET + section_0_y_offset;
        let section_title_height = 40.0;
        let content_area = Rect::new(
            container_rect.x,
            section_y + section_title_height,
            container_rect.width,
            240.0,
        );
        let item_rects = layout::stack_vertical(&content_area, &item_heights, 10.0, 0.0);

        self.provider_selector_rect = item_rects.get(1).copied().unwrap_or(Rect::new(0.0, 0.0, 0.0, 0.0));

        if let Some(rect) = item_rects.get(3) {
            self.model_id_input.position = rect.position();
            self.model_id_input.size = rect.size();
            self.openai_model_input.position = rect.position();
            self.openai_model_input.size = rect.size();
        }
        if let Some(hf_i) = hf_idx {
            if let Some(rect) = item_rects.get(hf_i) {
                self.hf_token_input.position = rect.position();
                self.hf_token_input.size = rect.size();
            }
        }

        // "Manage system prompts" button — SectionStack must match gfx/components/settings.rs render_settings.
        let mut full_stack = SectionStack::new(SECTION_SPACING);
        let mut model_sec = Section::new("Model Settings".to_string(), 40.0);
        model_sec.item_count = if provider == "openai" { 5 } else { 6 };
        model_sec.title_height = 40.0;
        full_stack.add_section(model_sec);
        let mut gen_sec = Section::new("Generation Settings".to_string(), 140.0);
        gen_sec.item_count = 1;
        gen_sec.title_height = 40.0;
        full_stack.add_section(gen_sec);
        let mut pers_sec = Section::new("Personalization".to_string(), 40.0);
        pers_sec.item_count = 2;
        pers_sec.title_height = 40.0;
        full_stack.add_section(pers_sec);
        let layout_full = full_stack.layout(&container_rect);
        let title_offset = TITLE_OFFSET;
        let gen_y = title_offset + layout_full.iter().find(|(i, _)| *i == 1).unwrap().1;
        let content_gen = full_stack.sections[1].content_rect(&container_rect, gen_y);
        self.manage_system_prompts_button.position = Vec2::new(content_gen.x, content_gen.y + 78.0);
        self.manage_system_prompts_button.size = Vec2::new(240.0, 36.0);
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }
}

