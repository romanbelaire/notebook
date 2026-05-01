//! System clipboard read/write (arboard).

pub fn set_text(text: &str) {
    let Ok(mut clipboard) = arboard::Clipboard::new() else {
        eprintln!("clipboard: failed to open clipboard for write");
        return;
    };
    if let Err(e) = clipboard.set_text(text) {
        eprintln!("clipboard: set_text: {}", e);
    }
}

pub fn get_text() -> String {
    let Ok(mut clipboard) = arboard::Clipboard::new() else {
        eprintln!("clipboard: failed to open clipboard for read");
        return String::new();
    };
    match clipboard.get_text() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("clipboard: get_text: {}", e);
            String::new()
        }
    }
}
