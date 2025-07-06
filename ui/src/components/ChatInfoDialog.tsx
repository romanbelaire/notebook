import { useInsightsStore } from "../store/insights";
import { useChatStore } from "../store/chat";
import type { Insight } from "../api";
import { useEffect, useState, useMemo } from "react";
import CloseIcon from "../assets/close.svg?react";
import PencilIcon from "../assets/pencil.svg?react";
import MagnifyIcon from "../assets/magnify.svg?react";
import PdfModal from "./PdfModal";

interface Props {
  conversationId: string;
  onClose: () => void;
}

export default function ChatInfoDialog({ conversationId, onClose }: Props) {
  const { conversations, deleteConversation, updateConversationTitle } = useChatStore((s) => ({
    conversations: s.conversations,
    deleteConversation: s.deleteConversation,
    updateConversationTitle: s.updateConversationTitle,
  }));
  const convo = conversations.find((c) => c.id === conversationId);
  const { insights, setModalInsight } = useInsightsStore();

  const [citations, setCitations] = useState<Record<string, unknown>[]>([]);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [draftTitle, setDraftTitle] = useState(convo ? convo.title : "");

  const [selectedInsight, setSelectedInsight] = useState<Insight | null>(null);

  // PDF viewer state
  const [pdfViewer, setPdfViewer] = useState<{ filename: string; page?: number } | null>(null);

  // Citation display mode: "all" shows every citation entry, "unique" collapses by source
  const citationTabs = ["all", "unique"] as const;
  type CitationMode = typeof citationTabs[number];
  const [citationMode, setCitationMode] = useState<CitationMode>("all");

  // Derive list based on selected mode
  const displayedCitations = useMemo(() => {
    if (citationMode === "all") return citations;
    const seen = new Set<string>();
    const uniques: Record<string, unknown>[] = [];
    for (const c of citations) {
      const key = (c as any).source ?? (c as any).title ?? JSON.stringify(c);
      if (!seen.has(key)) {
        seen.add(key);
        uniques.push(c);
      }
    }
    return uniques;
  }, [citations, citationMode]);

  useEffect(() => {
    if (!convo) return;
    const cites: Record<string, unknown>[] = [];
    for (const msg of convo.messages) {
      if (msg.role === "assistant" && msg.citations) {
        for (const c of msg.citations) {
          // Simple uniqueness by JSON string
          if (!cites.some((x) => JSON.stringify(x) === JSON.stringify(c))) {
            cites.push(c);
          }
        }
      }
    }
    setCitations(cites);
  }, [convo]);

  if (!convo) return null;

  // Match insights belonging to this conversation by message text match
  const convoTexts = new Set(convo.messages.map((m) => m.content));
  const convoInsights = insights.filter((i) => convoTexts.has(i.text));

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-primaryBg text-defaultText w-full max-w-lg p-6 rounded shadow-lg overflow-y-auto max-h-[90vh] relative">

        {/* If an insight is selected, render its modal */}
        {selectedInsight ? (
          <>
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-lg font-semibold truncate flex-1 min-w-0">
                {selectedInsight.title || "Insight"}
              </h3>
              <button
                className="w-6 h-6 p-0 flex items-center justify-center ml-2"
                onClick={() => setSelectedInsight(null)}
                aria-label="Close insight"
              >
                <CloseIcon className="w-4 h-4 flex-none" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium mb-1">Content:</h4>
                <p className="text-sm whitespace-pre-wrap break-words bg-gray-50 dark:bg-gray-700 p-3 rounded">
                  {selectedInsight.text}
                </p>
              </div>

              {selectedInsight.contexts && selectedInsight.contexts.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-1">Contexts:</h4>
                  <ul className="list-disc ml-5 space-y-1 text-xs">
                    {selectedInsight.contexts.map((ctx, idx) => (
                      <li key={idx}>{ctx}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="text-xs text-gray-500 dark:text-gray-400">
                Created: {new Date(selectedInsight.created_at).toLocaleString()}
              </div>
            </div>
          </>
        ) : (
          <>
        {/* Header with editable title */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            {isEditingTitle ? (
              <input
                className="text-xl font-semibold bg-transparent border-b border-trim focus:outline-none flex-1 min-w-0"
                value={draftTitle}
                autoFocus
                onChange={(e) => setDraftTitle(e.target.value)}
                onBlur={() => {
                  const newTitle = draftTitle.trim();
                  if (newTitle && newTitle !== convo.title) {
                    updateConversationTitle(convo.id, newTitle);
                  }
                  setIsEditingTitle(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    (e.target as HTMLInputElement).blur();
                  } else if (e.key === "Escape") {
                    setIsEditingTitle(false);
                    setDraftTitle(convo.title);
                  }
                }}
              />
            ) : (
              <>
                <h2 className="text-xl font-semibold truncate flex-1 min-w-0">{convo.title}</h2>
                <button
                  className="w-6 h-6 p-0 flex items-center justify-center bg-buttonBg/80 hover:bg-buttonBg rounded flex-none"
                  title="Edit title"
                  onClick={() => setIsEditingTitle(true)}
                >
                  <PencilIcon className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
          <button className="w-6 h-6 p-0 flex items-center justify-center ml-2" onClick={onClose} aria-label="Close">
            <CloseIcon className="w-4 h-4 flex-none" />
          </button>
        </div>

        <section className="mb-4">
          <div className="flex items-center mb-2 gap-3">
            <h3 className="font-semibold">Citations ({displayedCitations.length})</h3>
            {/* Mode slider */}
            <div
              className="relative grid bg-secondaryBg/60 rounded-full overflow-hidden h-6 w-32 text-xs"
              style={{ gridTemplateColumns: `repeat(${citationTabs.length}, 1fr)` }}
            >
              <span
                className="absolute inset-0 m-0.5 bg-buttonBg rounded-full transition-transform duration-300"
                style={{ width: `${100 / citationTabs.length}%`, transform: `translateX(${citationTabs.indexOf(citationMode) * 100}%)` }}
              />
              {citationTabs.map((t) => (
                <button
                  key={t}
                  className={
                    "relative z-10 w-full h-full flex items-center justify-center bg-transparent border-none focus:outline-none transition-colors" +
                    (t === citationMode ? " text-accentText" : " text-light/80 hover:text-light")
                  }
                  onClick={() => setCitationMode(t)}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>
          {displayedCitations.length === 0 ? (
            <p className="text-sm text-defaultText/70">No citations collected.</p>
          ) : (
            <ul className="list-disc ml-5 space-y-1 text-sm">
              {displayedCitations.map((c, idx) => {
                const title = (c as any).title as string | undefined;
                const source = (c as any).source as string | undefined;
                const year = (c as any).year as string | undefined;
                const section = (c as any).section as string | undefined;
                const page = (c as any).page as number | undefined;
                let display = title ? `${title} (${source ?? ""}${year ? ", " + year : ""})` : source ?? "";
                if (section) display += ` – ${section}`;
                if (page) display += `, p.${page}`;
                return (
                  <li key={idx}>
                    <span className="flex-1">{display}</span>
                    {source && (
                      <button
                        className="inline-flex w-4 h-4 ml-1 p-0 align-baseline items-center justify-center translate-y-1"
                        onClick={() => setPdfViewer({ filename: source, page })}
                        title="Preview"
                      >
                        <MagnifyIcon className="w-4 h-4" />
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="mb-4">
          <h3 className="font-semibold mb-2">Pinned Insights ({convoInsights.length})</h3>
          {convoInsights.length === 0 ? (
            <p className="text-sm text-defaultText/70">No pinned insights from this chat.</p>
          ) : (
            <ul className="list-disc ml-5 space-y-1 text-sm">
              {convoInsights.map((i) => (
                <li key={i.id} className="truncate">
                  <button
                    className="text-left w-full truncate hover:text-accentText focus:outline-none bg-transparent border-none p-0"
                    onClick={() => {
                      setModalInsight(i);
                      onClose();
                    }}
                    title={i.title ?? i.text}
                  >
                    { (i.title ?? i.text).slice(0,60) }{ (i.title ?? i.text).length > 60 ? '…' : '' }
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <div className="flex justify-end gap-2">
          <button
            className="px-3 py-1 bg-[#db363c] text-white rounded hover:opacity-90"
            onClick={() => {
              if (confirm("Delete this chat history?")) {
                deleteConversation(conversationId);
                onClose();
              }
            }}
          >
            Delete Chat
          </button>
          <button className="px-3 py-1 bg-buttonBg rounded" onClick={onClose}>Close</button>
        </div>
          </>
        )}

        {pdfViewer && (
          <PdfModal
            filename={pdfViewer.filename}
            initialPage={pdfViewer.page}
            onClose={() => setPdfViewer(null)}
          />
        )}
      </div>
    </div>
  );
} 