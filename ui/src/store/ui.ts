import { create } from "zustand";

// Types for available tabs in the application
export type AppTab = "Chat" | "Notepad" | "Library" | "Data" | "Settings";

interface UIState {
  activeTab: AppTab;
  setActiveTab: (tab: AppTab) => void;
  /**
   * When true, LibraryView should focus the "Create Collection" input and then
   * reset this flag via clearFocusNewCollection().
   */
  focusNewCollection: boolean;
  requestFocusNewCollection: () => void;
  clearFocusNewCollection: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  activeTab: "Chat",
  setActiveTab: (tab) => set({ activeTab: tab }),
  focusNewCollection: false,
  requestFocusNewCollection: () => set({ focusNewCollection: true }),
  clearFocusNewCollection: () => set({ focusNewCollection: false }),
})); 