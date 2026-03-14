use crate::stylus::Document;
use crate::persistence::{get_data_subdir};
use std::path::PathBuf;
use std::fs;
use serde_json;

pub struct DocumentPersistence;

impl DocumentPersistence {
    /// Get the documents directory path
    fn get_documents_dir() -> Result<PathBuf, std::io::Error> {
        get_data_subdir("documents")
    }

    /// Get the file path for a document by ID
    fn get_document_path(id: &str) -> Result<PathBuf, std::io::Error> {
        let dir = Self::get_documents_dir()?;
        Ok(dir.join(format!("{}.json", id)))
    }

    /// Save a document to disk
    pub fn save_document(id: &str, document: &Document) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::get_document_path(id)?;
        let json = serde_json::to_string_pretty(document)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Load a document from disk
    pub fn load_document(id: &str) -> Result<Document, Box<dyn std::error::Error>> {
        let path = Self::get_document_path(id)?;
        if !path.exists() {
            return Err(format!("Document {} not found", id).into());
        }
        let json = fs::read_to_string(&path)?;
        let document: Document = serde_json::from_str(&json)?;
        Ok(document)
    }

    /// Delete a document from disk
    pub fn delete_document(id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::get_document_path(id)?;
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// List all document IDs
    pub fn list_documents() -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let dir = Self::get_documents_dir()?;
        let mut ids = Vec::new();
        
        if !dir.exists() {
            return Ok(ids);
        }

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(stem.to_string());
                }
            }
        }
        
        Ok(ids)
    }

    /// Check if a document exists
    pub fn document_exists(id: &str) -> bool {
        Self::get_document_path(id)
            .map(|p| p.exists())
            .unwrap_or(false)
    }
}

