import { useEffect, useState } from "react";
import { listModels, checkModelAvailable, saveApiKey, listSystemPrompts, createSystemPrompt, updateSystemPrompt, selectSystemPrompt, deleteSystemPrompt } from "../api";
import { useSettingsStore } from "../store/settings";
import type { Provider, ThemeId } from "../store/settings";
import type { PromptsResponse } from "../api";
import TrashIcon from "../assets/trash.svg?react";
import CloseIcon from "../assets/close.svg?react";
import QuestionIcon from "../assets/question.svg?react";

export default function SettingsView() {
  const {
    provider,
    modelId,
    availableModels,
    openAIKey,
    claudeKey,
    hfKey,
    theme,
    systemPrompts,
    selectedPromptId,
    newTokensLimit,
    topKRetrievals,
    setProvider,
    setModelId,
    setAvailableModels,
    setOpenAIKey,
    setClaudeKey,
    setHfKey,
    setTheme,
    setSystemPrompts,
    setSelectedPromptId,
    setNewTokensLimit,
    setTopKRetrievals,
  } = useSettingsStore();

  const [searchTerm, setSearchTerm] = useState("");
  const [searchMsg, setSearchMsg] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [promptName, setPromptName] = useState("");
  const [promptContent, setPromptContent] = useState("");
  const [promptMsg, setPromptMsg] = useState<string | null>(null);
  const [promptDeleteConfirm, setPromptDeleteConfirm] = useState(false);

  const themes: { id: ThemeId; label: string }[] = [
    { id: "standard", label: "Standard (Dark Blue)" },
    { id: "sakura-light", label: "Sakura Light" },
    { id: "springtime-light", label: "Springtime Light" },
    { id: "forest-dark", label: "Forest Dark" },
    { id: "toadstool-light", label: "Toadstool Light" },
    { id: "acorn-dark", label: "Acorn Dark" },
    { id: "light", label: "Basic Light" },
    { id: "dark", label: "Dark (High Contrast)" },
  ];

  // Load models list when provider is local
  useEffect(() => {
    if (provider !== "local") return;
    (async () => {
      try {
        const models = await listModels();
        setAvailableModels(models);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("Failed to fetch models for local provider", err);
      }
    })();
  }, [provider, setAvailableModels]);

  // Load system prompts once
  useEffect(() => {
    (async () => {
      try {
        const data: PromptsResponse = await listSystemPrompts();
        setSystemPrompts(data.prompts);
        setSelectedPromptId(data.selected_id);
        const sel = data.prompts.find((p) => p.id === data.selected_id);
        if (sel) {
          setPromptName(sel.name);
          setPromptContent(sel.content);
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("Failed to load system prompts", err);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSearch = async () => {
    const name = searchTerm.trim();
    if (!name) return;
    setChecking(true);
    try {
      const res = await checkModelAvailable(name);
      if (res.available) {
        setSearchMsg("✅ Model available" + (res.gated ? " (gated)" : ""));
        if (!availableModels.includes(name)) {
          setAvailableModels([...availableModels, name]);
        }
      } else {
        setSearchMsg("❌ Not found, or gated");
      }
    } catch (err) {
      setSearchMsg("❌ Error checking model");
    } finally {
      setChecking(false);
    }
  };

  const handleApiSave = async (prov: "openai" | "claude" | "hf", key: string) => {
    try {
      await saveApiKey(prov, key);
      if (prov === "openai") {
        setOpenAIKey(key);
      } else if (prov === "claude") {
        setClaudeKey(key);
      } else {
        setHfKey(key);
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to save key", err);
    }
  };

  const handlePromptSelect = async (id: string) => {
    setSelectedPromptId(id);
    const sel = systemPrompts.find((p) => p.id === id);
    if (sel) {
      setPromptName(sel.name);
      setPromptContent(sel.content);
    }
    try {
      await selectSystemPrompt(id as string);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to select prompt", err);
    }
  };

  const handlePromptSave = async () => {
    const name = promptName.trim();
    const content = promptContent.trim();
    if (!name || !content) return;
    try {
      let id = selectedPromptId;
      if (id) {
        await updateSystemPrompt(id, name, content);
      } else {
        const created = await createSystemPrompt(name, content);
        id = created.id;
      }
      const data = await listSystemPrompts();
      setSystemPrompts(data.prompts);
      setSelectedPromptId(id);
      setPromptMsg("Saved");
      setTimeout(() => setPromptMsg(null), 2000);
      await selectSystemPrompt(id as string);
    } catch (err) {
      setPromptMsg("Error saving prompt");
    }
  };

  const handleNewPrompt = () => {
    setSelectedPromptId(null);
    setPromptName("");
    setPromptContent("");
  };

  const handlePromptDelete = async () => {
    if (!selectedPromptId) {
      setPromptDeleteConfirm(false);
      return;
    }
    try {
      await deleteSystemPrompt(selectedPromptId);
      const data = await listSystemPrompts();
      setSystemPrompts(data.prompts);
      setSelectedPromptId(data.selected_id);
      const sel = data.prompts.find((p) => p.id === data.selected_id);
      setPromptName(sel ? sel.name : "");
      setPromptContent(sel ? sel.content : "");
    } catch (err) {
      setPromptMsg("Error deleting prompt");
    } finally {
      setPromptDeleteConfirm(false);
    }
  };

  // Settings section tabs
  const sectionTabs = ["Model", "Generation", "Personalization"] as const;
  type Section = typeof sectionTabs[number];
  const [activeSection, setActiveSection] = useState<Section>("Model");

  return (
    <div className="w-full h-full overflow-auto p-6 bg-primaryBg text-defaultText">
      <h2 className="text-2xl font-semibold mb-6">Settings</h2>

      {/* Section slider */}
      <div className="mb-8 max-w-xl">
        <div
          className="relative grid bg-secondaryBg/60 rounded-full overflow-hidden h-10 shadow-inner"
          style={{ gridTemplateColumns: `repeat(${sectionTabs.length}, 1fr)` }}
        >
          <span
            className="absolute inset-0 m-0.5 bg-buttonBg rounded-full transition-transform duration-300 shadow"
            style={{ width: `${100 / sectionTabs.length}%`, transform: `translateX(${sectionTabs.indexOf(activeSection) * 100}%)` }}
          />
          {sectionTabs.map((t) => (
            <button
              key={t}
              className={
                "relative z-10 w-full h-full flex items-center justify-center text-sm bg-transparent border-none focus:outline-none transition-colors" +
                (t === activeSection ? " text-accentText" : " text-light/80 hover:text-light")
              }
              onClick={() => setActiveSection(t)}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* ───────────────────────────────────────── Active Section ── */}
      {activeSection === "Model" && (
        <>
          {/* Provider selection */}
          <div className="mb-6 max-w-lg">
            <label className="block font-medium mb-1">Backend Provider</label>
            <select
              className="w-full border border-trim bg-[var(--color-header-bg)] rounded px-2 py-1 text-defaultText"
              value={provider}
              onChange={(e) => setProvider(e.target.value as Provider)}
            >
              <option value="local">Local Model</option>
              {openAIKey && <option value="openai">OpenAI</option>}
              {claudeKey && <option value="claude">Claude</option>}
            </select>
          </div>

          {/* Model selection (only for local provider) */}
          {provider === "local" && (
            <div className="mb-6 max-w-lg">
              <label className="block font-medium mb-1">Select Model</label>
              <select
                className="w-full border border-trim bg-[var(--color-header-bg)] rounded px-2 py-1 text-defaultText"
                value={modelId ?? ""}
                onChange={(e) => setModelId(e.target.value)}
              >
                <option value="" disabled>
                  -- choose --
                </option>
                {availableModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>

              {/* Model search */}
              <div className="flex mt-2 gap-2">
                <input
                  className="flex-1 border border-trim rounded px-2 py-1 bg-[var(--color-header-bg)]"
                  placeholder="Search HuggingFace model..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                <button
                  className="px-3 py-1 border border-buttonBg bg-buttonBg rounded disabled:opacity-50"
                  onClick={handleSearch}
                  disabled={checking || !searchTerm.trim()}
                >
                  {checking ? "Checking..." : "Search"}
                </button>
              </div>
              {searchMsg && <p className="text-sm mt-1">{searchMsg}</p>}
            </div>
          )}

          {/* API keys section */}
          <div className="mb-6 space-y-4 max-w-lg">
            <div>
              <label className="block font-medium mb-1">OpenAI API Key</label>
              <input
                className="w-full border border-trim rounded px-2 py-1 bg-[var(--color-header-bg)]"
                type="password"
                placeholder="sk-..."
                defaultValue={openAIKey ?? ""}
                onBlur={(e) => handleApiSave("openai", e.target.value.trim())}
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Claude API Key</label>
              <input
                className="w-full border border-trim rounded px-2 py-1 bg-[var(--color-header-bg)]"
                type="password"
                placeholder="claude-..."
                defaultValue={claudeKey ?? ""}
                onBlur={(e) => handleApiSave("claude", e.target.value.trim())}
              />
            </div>
            <div>
              <label className="block font-medium mb-1">HuggingFace API Key</label>
              <input
                className="w-full border border-trim rounded px-2 py-1 bg-[var(--color-header-bg)]"
                type="password"
                placeholder="hf_..."
                defaultValue={hfKey ?? ""}
                onBlur={(e) => handleApiSave("hf", e.target.value.trim())}
              />
            </div>
          </div>
        </>
      )}

      {activeSection === "Generation" && (
        <>
          {/* System prompts */}
          <div className="mb-6 max-w-2xl">
            <label className="block font-medium mb-1 flex items-center gap-1">
              System Prompts
              <span title="Preset system prompt prepended to every user query to guide the model's behavior.">
                <QuestionIcon className="w-4 h-4 text-defaultText/70" />
              </span>
            </label>
            <div className="flex gap-2 items-center mb-2">
              <select
                className="flex-1 border border-trim bg-[var(--color-header-bg)] rounded px-2 py-1 text-defaultText"
                value={selectedPromptId ?? ""}
                onChange={(e) => handlePromptSelect(e.target.value)}
              >
                {systemPrompts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
              <button
                className="px-2 py-1 border border-buttonBg bg-buttonBg rounded"
                onClick={handleNewPrompt}
              >
                New
              </button>
            </div>
            <div className="flex items-center gap-2 mb-2">
              <input
                className="flex-1 border border-trim rounded px-2 py-1 bg-[var(--color-header-bg)]"
                placeholder="Prompt Name"
                value={promptName}
                onChange={(e) => setPromptName(e.target.value)}
              />
              {promptDeleteConfirm ? (
                <>
                  <button
                    className="flex px-2 py-1 h-5 items-center justify-center text-xs rounded bg-[#db363c]/50 hover:bg-[#db363c] focus:outline-none"
                    onClick={handlePromptDelete}
                  >
                    Delete
                  </button>
                  <button
                    className="w-5 h-5 p-0 flex items-center justify-center bg-transparent border-0 text-defaultText hover:bg-white/10 focus:outline-none"
                    onClick={() => setPromptDeleteConfirm(false)}
                  >
                    <CloseIcon className="w-4 h-4 pointer-events-none" />
                  </button>
                </>
              ) : (
                <button
                  className="w-5 h-5 p-0 flex items-center justify-center bg-transparent border-0 text-defaultText hover:bg-white/10 focus:outline-none disabled:opacity-50"
                  title="Delete system prompt"
                  disabled={systemPrompts.length <= 1}
                  onClick={() => setPromptDeleteConfirm(true)}
                >
                  <TrashIcon className="w-4 h-4 pointer-events-none" />
                </button>
              )}
            </div>
            <textarea
              className="w-full h-32 border border-trim bg-[var(--color-header-bg)] rounded px-2 py-1 text-defaultText mb-2"
              placeholder="Prompt Content"
              value={promptContent}
              onChange={(e) => setPromptContent(e.target.value)}
            />
            <button
              className="px-3 py-1 border border-buttonBg bg-buttonBg rounded disabled:opacity-50"
              onClick={handlePromptSave}
              disabled={!promptName.trim() || !promptContent.trim()}
            >
              Save
            </button>
            {promptMsg && <p className="text-sm mt-1">{promptMsg}</p>}
          </div>

          {/* Generation parameters */}
          <div className="mb-6 max-w-lg space-y-4">
            <div>
              <label className="block font-medium mb-1 flex items-center gap-1">
                Max New Tokens
                <span title="Maximum number of tokens the model can generate in its response.">
                  <QuestionIcon className="w-4 h-4 text-defaultText/70" />
                </span>
              </label>
              <input
                type="number"
                min={16}
                max={4096}
                value={newTokensLimit}
                onChange={(e) => setNewTokensLimit(Number(e.target.value))}
                className="w-full border border-trim rounded px-2 py-1 bg-[var(--color-header-bg)]"
              />
            </div>
            <div>
              <label className="block font-medium mb-1 flex items-center gap-1">
                Top-K Retrievals
                <span title="Number of top documents retrieved and provided as additional context for generation.">
                  <QuestionIcon className="w-4 h-4 text-defaultText/70" />
                </span>
              </label>
              <input
                type="number"
                min={1}
                max={20}
                value={topKRetrievals}
                onChange={(e) => setTopKRetrievals(Number(e.target.value))}
                className="w-full border border-trim rounded px-2 py-1 bg-[var(--color-header-bg)]"
              />
            </div>
          </div>
        </>
      )}

      {activeSection === "Personalization" && (
        <div className="mb-6 max-w-lg">
          <label className="block font-medium mb-1">Theme</label>
          <select
            className="w-full border border-trim bg-[var(--color-header-bg)] rounded px-2 py-1 text-defaultText"
            value={theme}
            onChange={(e) => setTheme(e.target.value as ThemeId)}
          >
            {themes.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
} 