use anyhow::{Context, Result};
use pdfium_render::prelude::Pdfium;
use std::path::Path;

fn try_bind(path: impl AsRef<Path>) -> Result<Pdfium> {
    let path = path.as_ref();
    let bindings = Pdfium::bind_to_library(path)
        .map_err(|e| anyhow::anyhow!("{}: {:?}", path.display(), e))?;
    Ok(Pdfium::new(bindings))
}

pub fn bind_pdfium() -> Result<Pdfium> {
    let mut attempted: Vec<String> = Vec::new();

    if let Ok(raw) = std::env::var("NOTEBOOK_PDFIUM_DLL") {
        attempted.push(format!("NOTEBOOK_PDFIUM_DLL={}", raw));
        if let Ok(p) = try_bind(raw.trim()) {
            return Ok(p);
        }
    }

    let exe = std::env::current_exe().context("resolve current_exe for PDFium")?;
    let beside_exe = Pdfium::pdfium_platform_library_name_at_path(exe.parent().unwrap());
    attempted.push(format!("beside_exe={}", beside_exe.display()));
    if let Ok(p) = try_bind(&beside_exe) {
        return Ok(p);
    }

    let crate_pdfium = Pdfium::pdfium_platform_library_name_at_path("pdfium");
    attempted.push(format!("pdfium_dir={}", crate_pdfium.display()));
    if let Ok(p) = try_bind(&crate_pdfium) {
        return Ok(p);
    }

    let cwd_lib = Pdfium::pdfium_platform_library_name_at_path(".");
    attempted.push(format!("cwd={}", cwd_lib.display()));
    if let Ok(p) = try_bind(&cwd_lib) {
        return Ok(p);
    }

    Pdfium::bind_to_system_library()
        .map(Pdfium::new)
        .with_context(|| {
            format!(
                "Could not load PDFium ({}).\nInstall pdfium.dll (Windows), put it next to notebook-native-ui.exe, in `./pdfium/`, your working directory, or on PATH; or set NOTEBOOK_PDFIUM_DLL to the DLL path.\nTried: {}",
                Pdfium::pdfium_platform_library_name().to_string_lossy(),
                attempted.join(", ")
            )
        })
}
