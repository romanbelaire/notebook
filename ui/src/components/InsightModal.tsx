import { useInsightsStore } from "../store/insights";
import { useChatStore } from "../store/chat";
import CloseIcon from "../assets/close.svg?react";
import PencilIcon from "../assets/pencil.svg?react";
import MagnifyIcon from "../assets/magnify.svg?react";
import { deleteInsight, updateInsight } from "../api";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { useState, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
// Import react-pdf CSS to fix TextLayer warning
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Configure pdf.js worker (same as PdfModal)
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

// Base backend URL for PDF serving
const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// PDF Peek Component
function PdfPeek({ source, page, onClose }: { source: string; page?: number; onClose: () => void }) {
  const pdfUrl = `${API_BASE}/papers/${source}`;
  
  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 z-40 bg-black/20"
        onClick={onClose}
      />
      {/* PDF Preview */}
      <div 
        className="fixed z-50 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl p-4 max-w-md"
        style={{ 
          left: '50%', 
          top: '50%', 
          transform: 'translate(-50%, -50%)',
          pointerEvents: 'auto'
        }}
        onMouseLeave={onClose}
        onMouseEnter={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium truncate">{source}</span>
          {page && <span className="text-xs text-gray-500 ml-2">p.{page}</span>}
        </div>
        <div className="w-64 h-80 overflow-hidden">
          <Document
            file={pdfUrl}
            loading={<div className="p-4 text-sm">Loading PDF preview...</div>}
            error={<div className="p-4 text-sm text-red-500">Failed to load PDF</div>}
          >
            <Page 
              pageNumber={page || 1} 
              width={256}
              className="border"
            />
          </Document>
        </div>
      </div>
    </>
  );
}

export default function InsightModal() {
  const { modalInsight, setModalInsight, updateInsightTitle, updateInsightText } = useInsightsStore((s) => ({
    modalInsight: s.modalInsight,
    setModalInsight: s.setModalInsight,
    updateInsightTitle: s.updateInsightTitle,
    updateInsightText: s.updateInsightText,
  }));

  const { conversations } = useChatStore();
  const queryClient = useQueryClient();

  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [draftTitle, setDraftTitle] = useState(modalInsight?.title ?? "");

  const [isEditingText, setIsEditingText] = useState(false);
  const [draftText, setDraftText] = useState(modalInsight?.text ?? "");

  const [pdfPeek, setPdfPeek] = useState<{ source: string; page?: number } | null>(null);

  // Place mutation hook before potential early return so the hooks order remains stable
  const deleteMutation = useMutation<void, Error, string>({
    mutationFn: deleteInsight,
    onSuccess: () => {
      // refresh list
      queryClient.invalidateQueries({ queryKey: ["insights"] });
      setModalInsight(null);
    },
  });

  const updateTextMutation = useMutation<void, Error, { id: string; text: string; title?: string }>({
    mutationFn: ({ id, text, title }) => updateInsight(id, text, title),
    onSuccess: (_, vars) => {
      updateInsightText(vars.id, vars.text);
      queryClient.invalidateQueries({ queryKey: ["insights"] });
      setIsEditingText(false);
    },
  });

  // keep draft title in sync when modal insight changes
  useEffect(() => {
    if (modalInsight && !isEditingTitle) {
      setDraftTitle(modalInsight.title);
    }
  }, [modalInsight?.title, isEditingTitle]);

  // keep draft text in sync when modal changes
  useEffect(() => {
    if (modalInsight && !isEditingText) {
      setDraftText(modalInsight.text);
    }
  }, [modalInsight?.text, isEditingText]);

  // Early return must come after hook declarations to maintain consistent order
  if (!modalInsight) return null;

  const html = (() => {
    try {
      const parsed = marked.parse(modalInsight.text);
      return typeof parsed === "string" ? DOMPurify.sanitize(parsed) : "";
    } catch {
      return modalInsight.text;
    }
  })();

  const formatCitation = (c: Record<string, unknown>) => {
    const title = (c as any).title as string | undefined;
    const source = (c as any).source as string | undefined;
    const year = (c as any).year as string | undefined;
    const section = (c as any).section as string | undefined;
    const page = (c as any).page as number | undefined;
    let display = title ? `${title} (${source ?? ""}${year ? ", " + year : ""})` : source ?? "";
    if (section) display += ` – ${section}`;
    if (page) display += `, p.${page}`;
    return display;
  };

  // locate matching citations
  const msg = conversations.flatMap((c) => c.messages).find((m) => m.content === modalInsight.text && m.citations && m.citations.length);

  // Build unique citation→context list to avoid duplicates
  const sourceItems = (() => {
    if (!msg || !msg.citations) return [] as { cit: Record<string, unknown>; ctx?: string }[];
    const items: { cit: Record<string, unknown>; ctx?: string }[] = [];
    msg.citations.forEach((c, idx) => {
      const key = JSON.stringify(c);
      if (!items.some((it) => JSON.stringify(it.cit) === key)) {
        items.push({ cit: c, ctx: modalInsight.contexts?.[idx] });
      }
    });
    return items;
  })();

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setModalInsight(null)}>
      <div className="bg-primaryBg text-defaultText w-full max-w-2xl p-6 rounded shadow-lg overflow-y-auto max-h-[90vh]" onClick={(e)=>e.stopPropagation()}>
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
                  if (newTitle && newTitle !== modalInsight.title) {
                    updateInsightTitle(modalInsight.id, newTitle);
                  }
                  setIsEditingTitle(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    (e.target as HTMLInputElement).blur();
                  } else if (e.key === "Escape") {
                    setIsEditingTitle(false);
                    setDraftTitle(modalInsight.title);
                  }
                }}
              />
            ) : (
              <>
                <h2 className="text-xl font-semibold truncate flex-1 min-w-0">{modalInsight.title || "Insight"}</h2>
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
          <button className="w-6 h-6 p-0 flex items-center justify-center ml-2" onClick={() => setModalInsight(null)} aria-label="Close">
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>
        <div className="space-y-4">
          {/* Content */}
          <div>
            <h4 className="text-sm font-medium mb-1">Content:</h4>
            {isEditingText ? (
              <>
                <textarea
                  className="w-full min-h-[10rem] bg-gray-50 dark:bg-gray-700 p-3 rounded text-defaultText"
                  value={draftText}
                  onChange={(e) => setDraftText(e.target.value)}
                />
                <div className="flex gap-2 mt-2">
                  <button
                    className="px-3 py-1 bg-buttonBg rounded disabled:opacity-50"
                    disabled={updateTextMutation.isPending || draftText.trim() === ""}
                    onClick={() => {
                      const newText = draftText.trim();
                      if (newText && newText !== modalInsight.text) {
                        updateTextMutation.mutate({ id: modalInsight.id, text: newText, title: modalInsight.title });
                      } else {
                        setIsEditingText(false);
                      }
                    }}
                  >
                    Save
                  </button>
                  <button
                    className="px-3 py-1 bg-gray-500/50 rounded"
                    onClick={() => {
                      setIsEditingText(false);
                      setDraftText(modalInsight.text);
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <div
                className="prose prose-invert max-w-none bg-gray-50 dark:bg-gray-700 p-3 rounded"
                dangerouslySetInnerHTML={{ __html: html }}
              />
            )}
            {!isEditingText && (
              <button
                className="mt-2 flex items-center gap-1 text-sm text-defaultText/80 hover:text-defaultText"
                onClick={() => setIsEditingText(true)}
              >
                <PencilIcon className="w-4 h-4" /> Edit
              </button>
            )}
          </div>

          {/* Source excerpts */}
          {sourceItems.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold">Sources</h3>
              {sourceItems.map(({ cit: c, ctx }, idx) => {
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center gap-1 relative">
                      <h4 className="text-sm font-medium truncate flex-1" title={formatCitation(c)}>
                        {formatCitation(c)}
                      </h4>
                      {(c as any).source && (
                        <div className="relative">
                          <button
                            className="inline-flex w-4 h-4 p-0 align-baseline items-center justify-center hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
                            onMouseEnter={(e) => {
                              console.log('Mouse enter on magnify button for:', (c as any).source);
                              setPdfPeek({ 
                                source: (c as any).source, 
                                page: (c as any).page 
                              });
                            }}
                            onMouseLeave={(e) => {
                              console.log('Mouse leave on magnify button');
                              // Add a small delay to prevent flickering
                              setTimeout(() => {
                                if (pdfPeek && pdfPeek.source === (c as any).source) {
                                  setPdfPeek(null);
                                }
                              }, 100);
                            }}
                            title="PDF Preview"
                          >
                            <MagnifyIcon className="w-4 h-4" />
                          </button>
                          {pdfPeek && pdfPeek.source === (c as any).source && (
                            <PdfPeek 
                              source={pdfPeek.source} 
                              page={pdfPeek.page}
                              onClose={() => {
                                console.log('Closing PDF peek');
                                setPdfPeek(null);
                              }}
                            />
                          )}
                        </div>
                      )}
                    </div>
                    {ctx && (
                      <p className="text-xs whitespace-pre-wrap bg-gray-50 dark:bg-gray-700 p-2 rounded">
                        {ctx}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Timestamp */}
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Created: {new Date(modalInsight.created_at).toLocaleString()}
          </div>
        </div>

        {/* Footer with delete & close */}
        <div className="flex justify-end gap-2 mt-6">
          <button
            className="px-3 py-1 bg-[#db363c] text-white rounded hover:opacity-90"
            onClick={() => {
              if (confirm("Delete this insight?")) {
                deleteMutation.mutate(modalInsight.id);
              }
            }}
          >
            Delete Insight
          </button>
          <button
            className="px-3 py-1 bg-buttonBg rounded"
            onClick={() => setModalInsight(null)}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
} 