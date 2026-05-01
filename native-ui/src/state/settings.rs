use serde::{Serialize, Deserialize};

/// User-defined prompt invoked from chat as `/name` (name is matched case-insensitively).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptEntry {
    pub id: String,
    pub name: String,
    pub content: String,
}

fn default_system_prompts() -> Vec<SystemPromptEntry> {
    Vec::new()
}

/// If input starts with `/token` and *token* matches a named prompt, returns (user message after command, system text).
/// Otherwise returns (trimmed original, None).
pub fn parse_slash_system_prompt(text: &str, prompts: &[SystemPromptEntry]) -> (String, Option<String>) {
    let t = text.trim();
    if !t.starts_with('/') {
        return (t.to_string(), None);
    }
    let after = &t[1..];
    let first_space = after.find(' ');
    let (token, user_rest) = match first_space {
        Some(i) => (&after[..i], after[i + 1..].trim_start()),
        None => (after, ""),
    };
    if token.is_empty() {
        return (t.to_string(), None);
    }
    let token_lower = token.to_lowercase();
    for p in prompts {
        if p.name.to_lowercase() == token_lower {
            return (user_rest.to_string(), Some(p.content.clone()));
        }
    }
    (t.to_string(), None)
}

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
    #[serde(default = "default_system_prompts")]
    pub system_prompts: Vec<SystemPromptEntry>,
}

impl SettingsState {
    pub fn new() -> Self {
        Self {
            hf_token: String::new(),
            model_id: default_local_model_id().to_string(),
            provider: default_provider(),
            openai_model: default_openai_model(),
            theme: "standard".to_string(),
            api_base_url: Some("http://localhost:8000".to_string()),
            system_prompts: Vec::new(),
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

