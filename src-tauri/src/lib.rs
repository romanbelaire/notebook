#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![validate_directory, pick_folder, browse_parent_folder, open_folder_in_explorer])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
fn validate_directory(path: String) -> Result<bool, String> {
    use std::path::Path;
    let dir_path = Path::new(&path);
    Ok(dir_path.exists() && dir_path.is_dir())
}

#[tauri::command]
async fn pick_folder(default_path: Option<String>) -> Result<Option<String>, String> {
    use rfd::AsyncFileDialog;
    
    // Follow Microsoft's recommended approach: FolderPicker with FileTypeFilter
    let mut dialog = AsyncFileDialog::new()
        .set_title("Select PDF Directory");
    
    // Add file type filters to show files in the folder picker (Microsoft's approach)
    // This should make the dialog show files while still being in folder selection mode
    dialog = dialog
        .add_filter("PDF Files", &["pdf"])
        .add_filter("Text Files", &["txt", "md"])
        .add_filter("Document Files", &["doc", "docx"])
        .add_filter("All Files", &["*"]);
    
    if let Some(default) = default_path {
        if let Ok(path) = std::path::Path::new(&default).canonicalize() {
            dialog = dialog.set_directory(path);
        }
    }
    
    // Use pick_folder() but with file type filters (following Microsoft's pattern)
    match dialog.pick_folder().await {
        Some(folder) => Ok(Some(folder.path().to_string_lossy().to_string())),
        None => Ok(None),
    }
}

#[tauri::command]
async fn browse_parent_folder(current_path: Option<String>) -> Result<Option<String>, String> {
    use rfd::AsyncFileDialog;
    
    // Try Windows-specific approach for better focus behavior
    #[cfg(target_os = "windows")]
    {
        // On Windows, try using file picker with a dummy file in the target directory
        // This should focus on the directory when we pick a file and return its parent
        let mut dialog = AsyncFileDialog::new()
            .set_title("Select any file in the target directory (the folder will be used)")
            .add_filter("PDF Files", &["pdf"])
            .add_filter("Text Files", &["txt", "md"]) 
            .add_filter("Document Files", &["doc", "docx"])
            .add_filter("All Files", &["*"]);
        
        if let Some(current) = current_path {
            let current_path = std::path::Path::new(&current);
            
            // Start in the target directory itself for file picker
            if let Ok(current_canonical) = current_path.canonicalize() {
                dialog = dialog.set_directory(current_canonical);
            } else if let Some(parent) = current_path.parent() {
                if let Ok(parent_canonical) = parent.canonicalize() {
                    dialog = dialog.set_directory(parent_canonical);
                }
            }
        }
        
        // Use file picker but return the directory containing the selected file
        match dialog.pick_file().await {
            Some(file_handle) => {
                let file_path = file_handle.path();
                if let Some(parent_dir) = file_path.parent() {
                    Ok(Some(parent_dir.to_string_lossy().to_string()))
                } else {
                    Ok(Some(file_path.to_string_lossy().to_string()))
                }
            }
            None => Ok(None),
        }
    }
    
    // Fallback for other platforms - standard folder picker
    #[cfg(not(target_os = "windows"))]
    {
        let mut dialog = AsyncFileDialog::new()
            .set_title("Browse to PDF Directory")
            .add_filter("PDF Files", &["pdf"])
            .add_filter("Text Files", &["txt", "md"])
            .add_filter("Document Files", &["doc", "docx"])
            .add_filter("All Files", &["*"]);
        
        if let Some(current) = current_path {
            let current_path = std::path::Path::new(&current);
            
            if let Some(parent) = current_path.parent() {
                if let Ok(parent_canonical) = parent.canonicalize() {
                    dialog = dialog.set_directory(parent_canonical);
                }
            } else if let Ok(current_canonical) = current_path.canonicalize() {
                dialog = dialog.set_directory(current_canonical);
            }
        }
        
        match dialog.pick_folder().await {
            Some(folder) => Ok(Some(folder.path().to_string_lossy().to_string())),
            None => Ok(None),
        }
    }
}

#[tauri::command]
async fn open_folder_in_explorer(path: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    
    let path_buf = std::path::Path::new(&path);
    
    if !path_buf.exists() {
        return Err(format!("Directory does not exist: {}", path));
    }
    
    if !path_buf.is_dir() {
        return Err(format!("Path is not a directory: {}", path));
    }
    
    // Open the folder in the native file browser (Explorer on Windows)
    tauri_plugin_opener::open_path(path_buf, None::<String>)
        .map_err(|e| format!("Failed to open folder: {}", e))?;
    
    Ok(())
}
