interface OpenDialogOptions {
  filters?: { name: string; extensions: string[] }[];
  multiple?: boolean;
}

/**
 * Browser fallback for Tauri's dialog.open API. Always shows an alert and
 * returns null so callers can handle the lack of selection gracefully.
 */
export async function open(_options?: OpenDialogOptions): Promise<string | null> {
  alert('File-open dialogs are only available in the desktop build.');
  return null;
}

/**
 * Simple stub for dialog.message – just forwards to window.alert.
 */
export async function message(msg: string, { title }: { title?: string } = {}): Promise<void> {
  alert(title ? `${title}\n\n${msg}` : msg);
} 