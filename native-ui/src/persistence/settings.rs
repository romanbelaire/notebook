use crate::state::settings::SettingsState;
use crate::persistence::get_data_dir;
use std::path::PathBuf;
use std::fs;
use serde_json;

pub struct SettingsPersistence;

impl SettingsPersistence {
    /// Get the settings file path
    fn get_settings_path() -> Result<PathBuf, std::io::Error> {
        let data_dir = get_data_dir()?;
        Ok(data_dir.join("settings.json"))
    }

    /// Save settings to disk
    pub fn save_settings(settings: &SettingsState) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::get_settings_path()?;
        let json = serde_json::to_string_pretty(settings)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Load settings from disk
    pub fn load_settings() -> Result<SettingsState, Box<dyn std::error::Error>> {
        let path = Self::get_settings_path()?;
        if !path.exists() {
            return Ok(SettingsState::new());
        }
        let json = fs::read_to_string(&path)?;
        let settings: SettingsState = serde_json::from_str(&json)?;
        Ok(settings)
    }
}

