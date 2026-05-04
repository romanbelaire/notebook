pub mod document;
pub mod conversation;
pub mod settings;
pub mod graph_layout;
pub mod pdf_annotations;
pub mod pdf_reading_positions;

pub use document::DocumentPersistence;
pub use conversation::ConversationPersistence;
pub use settings::SettingsPersistence;
pub use graph_layout::GraphLayoutPersistence;
pub use pdf_annotations::PdfAnnotationPersistence;
pub use pdf_reading_positions::{PdfReadingPositionPersistence, ReadingPosition};

use std::path::PathBuf;
use std::fs;

/// Get the base data directory for the application
/// Creates it if it doesn't exist
pub fn get_data_dir() -> Result<PathBuf, std::io::Error> {
    let data_dir = PathBuf::from("data");
    if !data_dir.exists() {
        fs::create_dir_all(&data_dir)?;
    }
    Ok(data_dir)
}

/// Get or create a subdirectory within the data directory
pub fn get_data_subdir(subdir: &str) -> Result<PathBuf, std::io::Error> {
    let data_dir = get_data_dir()?;
    let subdir_path = data_dir.join(subdir);
    if !subdir_path.exists() {
        fs::create_dir_all(&subdir_path)?;
    }
    Ok(subdir_path)
}

