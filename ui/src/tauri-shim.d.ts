declare module "@tauri-apps/api/shell" {
  export function open(path: string): Promise<void>;
} 