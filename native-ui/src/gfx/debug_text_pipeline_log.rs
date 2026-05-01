//! Append-only log for `NOTEBOOK_DEBUG_TEXT_PIPELINE` diagnostics (easier to share than stderr).
use std::io::Write;
use std::sync::{Mutex, OnceLock};

static LOG_FILE: OnceLock<Mutex<std::fs::File>> = OnceLock::new();
static LOG_PATH: OnceLock<String> = OnceLock::new();

fn log_path() -> &'static str {
    LOG_PATH.get_or_init(|| {
        std::env::var("NOTEBOOK_DEBUG_TEXT_LOG_PATH")
            .unwrap_or_else(|_| "data/debug_text_pipeline.log".to_string())
    })
}

/// Open/create log file once and append a line.
pub fn append_line(line: &str) {
    let path = log_path();
    let m = LOG_FILE.get_or_init(|| {
        let p = std::path::Path::new(path);
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).expect("debug_text_pipeline log parent dir");
        }
        let truncate = std::env::var("NOTEBOOK_DEBUG_TEXT_LOG_TRUNCATE")
            .map(|v| {
                let v = v.trim();
                v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
            })
            .unwrap_or(false);
        let f = if truncate {
            std::fs::File::create(p).expect("create debug_text_pipeline log (truncate)")
        } else {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("open NOTEBOOK_DEBUG_TEXT_LOG_PATH / data/debug_text_pipeline.log")
        };
        Mutex::new(f)
    });
    let mut g = m.lock().expect("debug log mutex");
    writeln!(g, "{line}").expect("debug log write");
}

pub fn append_session_header() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_millis();
    append_line(&format!("==== debug_text_pipeline session {ms} ===="));
}
