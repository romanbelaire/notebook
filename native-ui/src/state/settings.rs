use serde::{Serialize, Deserialize};

fn default_provider() -> String {
    "local".to_string()
}

fn default_openai_model() -> String {
    "gpt-4o".to_string()
}

pub fn default_local_model_id() -> &'static str {
    "meta-llama/Llama-3.2-1B-Instruct"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsState {
    pub hf_token: String,
    pub model_id: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_openai_model")]
    pub openai_model: String,
    pub theme: String,
    pub api_base_url: Option<String>,
}

impl SettingsState {
    pub fn new() -> Self {
        Self {
            hf_token: String::new(),
            model_id: default_local_model_id().to_string(),
            provider: default_provider(),
            openai_model: default_openai_model(),
            theme: "dark".to_string(),
            api_base_url: Some("http://localhost:8000".to_string()),
        }
    }

    pub fn update_hf_token(&mut self, token: String) {
        self.hf_token = token;
    }

    pub fn update_model_id(&mut self, model_id: String) {
        self.model_id = model_id;
    }

    pub fn update_theme(&mut self, theme: String) {
        self.theme = theme;
    }

    pub fn update_api_base_url(&mut self, url: Option<String>) {
        self.api_base_url = url;
    }

    pub fn update_provider(&mut self, provider: String) {
        self.provider = provider;
    }

    pub fn update_openai_model(&mut self, model: String) {
        self.openai_model = model;
    }

    /// For graph send: None when provider is openai; Some(model_id) for local, using default when empty.
    pub fn model_id_for_send(&self) -> Option<String> {
        if self.provider == "openai" {
            None
        } else {
            Some(if self.model_id.trim().is_empty() {
                default_local_model_id().to_string()
            } else {
                self.model_id.clone()
            })
        }
    }

    /// For graph send: None when provider is not openai; Some(openai_model) for openai.
    pub fn openai_model_for_send(&self) -> Option<String> {
        if self.provider == "openai" {
            Some(self.openai_model.clone())
        } else {
            None
        }
    }
}

impl Default for SettingsState {
    fn default() -> Self {
        Self::new()
    }
}

