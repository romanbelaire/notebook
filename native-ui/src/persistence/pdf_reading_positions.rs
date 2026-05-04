use crate::persistence::get_data_dir;
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadingPosition {
    pub page: u32,
    pub scroll_y: f32,
}

pub struct PdfReadingPositionPersistence;

impl PdfReadingPositionPersistence {
    fn path() -> Result<PathBuf, std::io::Error> {
        Ok(get_data_dir()?.join("pdf_reading_positions.json"))
    }

    fn load_all() -> Result<HashMap<String, ReadingPosition>, Box<dyn std::error::Error>> {
        let path = Self::path()?;
        if !path.exists() {
            return Ok(HashMap::new());
        }
        let json = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&json)?)
    }

    fn save_all(
        map: &HashMap<String, ReadingPosition>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::path()?;
        let json = serde_json::to_string_pretty(map)?;
        fs::write(&path, json)?;
        Ok(())
    }

    pub fn load(filename: &str) -> Option<ReadingPosition> {
        Self::load_all().unwrap_or_default().remove(filename)
    }

    pub fn save(filename: &str, pos: ReadingPosition) {
        let mut map = Self::load_all().unwrap_or_default();
        map.insert(filename.to_string(), pos);
        if let Err(e) = Self::save_all(&map) {
            eprintln!("Failed to save PDF reading position: {}", e);
        }
    }
}
