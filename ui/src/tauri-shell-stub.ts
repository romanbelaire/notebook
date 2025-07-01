export async function open(path: string): Promise<void> {
  // Browser fallback when Tauri's shell API is unavailable.
  if (typeof window === "undefined") return;

  // Browsers block local file URIs. Provide a helpful message instead of
  // spamming the console with security errors.
  if (path.startsWith("file://")) {
    console.warn("Cannot open local file system path in browser sandbox:", path);
    alert("Opening local folders is only supported in the desktop build. Please navigate to the 'data/papers' directory manually.");
    return;
  }

  try {
    window.open(path, "_blank", "noopener,noreferrer");
  } catch (err) {
    console.error(err);
    alert(`Failed to open path: ${String(err)}`);
    throw err;
  }
} 