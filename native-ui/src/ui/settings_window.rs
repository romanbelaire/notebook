use glam::Vec2;
use crate::ui::core::Rect;
use crate::ui::{ScrollView, TextInput, VStack};

pub struct SettingsWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub scroll_view: ScrollView,
    pub hf_token_input: TextInput,
    pub model_id_input: TextInput,
    pub openai_model_input: TextInput,
    pub selected_theme: usize,
    /// Clickable rect for the provider row (toggle Local / OpenAI)
    pub provider_selector_rect: Rect,
    // Persistent VStacks for each section (avoid recreating each frame)
    pub model_settings_stack: VStack,
    pub generation_settings_stack: VStack,
    pub personalization_stack: VStack,
}

impl SettingsWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
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
        model_settings_stack.add_text_styled("Provider:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
        model_settings_stack.add_text_styled("Local model", style::font_size::NORMAL, style::text::PRIMARY, TextAlignment::Left);
        model_settings_stack.add_text_styled("Model ID:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
        model_settings_stack.add_text_styled("HuggingFace API Key:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
        
        let mut generation_settings_stack = VStack::new(10.0, 0.0);
        generation_settings_stack.add_text_styled("System Prompts:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
        generation_settings_stack.add_text_styled("System prompt management coming soon...", style::font_size::SMALL, style::text::SECONDARY, TextAlignment::Left);
        
        let mut personalization_stack = VStack::new(10.0, 0.0);
        personalization_stack.add_text_styled("Theme:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
        personalization_stack.add_text_styled("Standard (Dark Blue)", style::font_size::NORMAL, style::text::PRIMARY, TextAlignment::Left);
        
        Self {
            position,
            size,
            scroll_view,
            hf_token_input,
            model_id_input,
            openai_model_input,
            selected_theme: 0,
            provider_selector_rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            model_settings_stack,
            generation_settings_stack,
            personalization_stack,
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
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }
}

