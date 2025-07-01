import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Provider = "local" | "openai" | "claude";

export type ThemeId =
  | "standard"
  | "sakura-light"
  | "springtime-light"
  | "forest-dark"
  | "toadstool-light"
  | "acorn-dark"
  | "light"
  | "dark";

interface SettingsState {
  provider: Provider;
  modelId: string | null;
  availableModels: string[];
  openAIKey: string | null;
  claudeKey: string | null;
  hfKey: string | null;
  newTokensLimit: number;
  topKRetrievals: number;
  theme: ThemeId;
  systemPrompts: { id: string; name: string; content: string }[];
  selectedPromptId: string | null;
  // actions
  setProvider: (p: Provider) => void;
  setModelId: (id: string | null) => void;
  setAvailableModels: (list: string[]) => void;
  setOpenAIKey: (key: string | null) => void;
  setClaudeKey: (key: string | null) => void;
  setHfKey: (key: string | null) => void;
  setNewTokensLimit: (limit: number) => void;
  setTopKRetrievals: (k: number) => void;
  setTheme: (theme: ThemeId) => void;
  setSystemPrompts: (prompts: { id: string; name: string; content: string }[]) => void;
  setSelectedPromptId: (id: string | null) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      provider: "local",
      modelId: null,
      availableModels: [],
      openAIKey: null,
      claudeKey: null,
      hfKey: null,
      newTokensLimit: 512,
      topKRetrievals: 5,
      theme: "standard",
      systemPrompts: [],
      selectedPromptId: null,
      setProvider: (provider) => set({ provider }),
      setModelId: (id) => set({ modelId: id }),
      setAvailableModels: (list) => set({ availableModels: list }),
      setOpenAIKey: (key) => set({ openAIKey: key }),
      setClaudeKey: (key) => set({ claudeKey: key }),
      setHfKey: (key) => set({ hfKey: key }),
      setNewTokensLimit: (limit) => set({ newTokensLimit: limit }),
      setTopKRetrievals: (k) => set({ topKRetrievals: k }),
      setTheme: (theme) => set({ theme }),
      setSystemPrompts: (prompts) => set({ systemPrompts: prompts }),
      setSelectedPromptId: (id) => set({ selectedPromptId: id }),
    }),
    { name: "notebook-settings" }
  )
);

// Some bundlers may fail to track `export const` in certain HMR edge-cases. Provide explicit alias and default export.
export { useSettingsStore as default }; 