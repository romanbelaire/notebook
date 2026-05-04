use crate::persistence::get_data_dir;
use crate::ui::pdf_modal::PdfAnnotation;
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;
use serde_json;

pub struct PdfAnnotationPersistence;

impl PdfAnnotationPersistence {
    fn path() -> Result<PathBuf, std::io::Error> {
        Ok(get_data_dir()?.join("pdf_annotations.json"))
    }

    fn load_all() -> Result<HashMap<String, Vec<PdfAnnotation>>, Box<dyn std::error::Error>> {
        let path = Self::path()?;
        if !path.exists() {
            return Ok(HashMap::new());
        }
        let json = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&json)?)
    }

    fn save_all(
        map: &HashMap<String, Vec<PdfAnnotation>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::path()?;
        let json = serde_json::to_string_pretty(map)?;
        fs::write(&path, json)?;
        Ok(())
    }

    pub fn load(filename: &str) -> Vec<PdfAnnotation> {
        Self::load_all()
            .unwrap_or_default()
            .remove(filename)
            .unwrap_or_default()
    }

    pub fn save(filename: &str, annotations: &[PdfAnnotation]) {
        let mut map = Self::load_all().unwrap_or_default();
        if annotations.is_empty() {
            map.remove(filename);
        } else {
            map.insert(filename.to_string(), annotations.to_vec());
        }
        if let Err(e) = Self::save_all(&map) {
            eprintln!("Failed to save PDF annotations: {}", e);
        }
    }
}
