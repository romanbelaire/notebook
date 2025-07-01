import { useRef, useState, useEffect } from "react";
import type { ChatMessage } from "../store/chat";
import { useChatStore } from "../store/chat";
import { postChat, createInsight, deleteInsight, listCollections, setContextPool } from "../api";
import { useInsightsStore } from "../store/insights";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import DOMPurify from "dompurify";
import { marked } from "marked";
// @ts-ignore – mark.js has no bundled types
import Mark from "mark.js";
import PdfModal from "./PdfModal";
import { useSettingsStore } from "../store/settings";
import squancePng from "../assets/squance.png";
import PinIcon from "../assets/pin.svg?react";
import PinRedIcon from "../assets/pin-red.svg?react";
import TrashIcon from "../assets/trash.svg?react";
import EyeOpenIcon from "../assets/eye-open.svg?react";
import EyeClosedIcon from "../assets/eye-closed.svg?react";
import MagnifyIcon from "../assets/magnify.svg?react";
import BookIcon from "../assets/book.svg?react";
import PencilIcon from "../assets/pencil.svg?react";
import CloseIcon from "../assets/close.svg?react";
import { clsx } from "clsx";
import type { Collection } from "../api";
import PlusIcon from "../assets/plus.svg?react";
import { useUIStore } from "../store/ui";

export default function ChatWindow() {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const {
    history,
    addMessage,
    isSending,
    setSending,
    activeId,
    finalizeConversation,
    updateMessage,
    deleteMessage,
  } = useChatStore((state) => ({
    history: state.history,
    addMessage: state.addMessage,
    isSending: state.isSending,
    setSending: state.setSending,
    activeId: state.activeId,
    finalizeConversation: state.finalizeConversation,
    updateMessage: state.updateMessage,
    deleteMessage: state.deleteMessage,
  }));
  const { insights, removeInsight } = useInsightsStore();
  const [input, setInput] = useState("");
  const queryClient = useQueryClient();
  const [highlightTerm, setHighlightTerm] = useState("");
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  // Mark.js instance (stored in ref to persist across renders)
  const markInstance = useRef<any>(null);
  // PDF viewer state
  const [pdfViewer, setPdfViewer] = useState<{ filename: string; page?: number } | null>(null);

  // Settings
  const { provider, modelId } = useSettingsStore();

  // Context pool dropdown state
  const [ctxDropdownOpen, setCtxDropdownOpen] = useState(false);
  const { data: collections } = useQuery<Collection[]>({ queryKey: ["collections"], queryFn: listCollections });
  const [selectedCollectionId, setSelectedCollectionId] = useState<number | null>(null);

  // UI store actions for tab switching and focusing the create-collection input
  const setActiveTab = useUIStore((s) => s.setActiveTab);
  const requestFocusNewCollection = useUIStore((s) => s.requestFocusNewCollection);

  const ctxPoolMutation = useMutation<void, Error, number | null>({
    mutationFn: (id) => setContextPool(id),
    onError: (err) => {
      alert(`Failed to set context pool: ${String(err)}`);
    },
  });

  const dropdownRef = useRef<HTMLDivElement | null>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setCtxDropdownOpen(false);
      }
    };
    if (ctxDropdownOpen) {
      window.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      window.removeEventListener("mousedown", handleClickOutside);
    };
  }, [ctxDropdownOpen]);

  const handleCtxPoolChange = (val: string) => {
    const id = val === "all" ? null : Number(val);
    setSelectedCollectionId(id);
    ctxPoolMutation.mutate(id);
    setCtxDropdownOpen(false);
  };

  // Initialize Mark.js when container ref becomes available
  useEffect(() => {
    if (messageContainerRef.current) {
      markInstance.current = new Mark(messageContainerRef.current);
    }
    return () => {
      // cleanup reference
      markInstance.current = null;
    };
  }, []);

  // Apply / clear highlights whenever highlightTerm changes
  useEffect(() => {
    if (!markInstance.current) return;
    // Always unmark first, then re-mark if term is present
    markInstance.current.unmark({
      done: () => {
        if (highlightTerm) {
          markInstance.current.mark(highlightTerm, {
            separateWordSearch: false,
            className: "highlight-term",
          });
        }
      },
    });
  }, [highlightTerm]);

  // Handler: when user selects text, capture it for highlighting
  const handleSelection = () => {
    const sel = window.getSelection();
    if (!sel) {
      setHighlightTerm("");
      return;
    }
    const text = sel.toString().trim();
    // Only keep reasonably short selections to avoid heavy DOM work
    if (text.length > 0 && text.length <= 60) {
      setHighlightTerm(text);
    } else {
      setHighlightTerm("");
    }
  };

  const addMutation = useMutation<{ id: string }, Error, ChatMessage>({
    mutationFn: async (msg: ChatMessage) => {
      const id = await createInsight(msg.content, msg.contexts ?? []);
      return { id };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["insights"] });
    },
  });

  const deleteMutation = useMutation<void, Error, { id: string }>({
    mutationFn: async ({ id }) => {
      await deleteInsight(id);
    },
    onSuccess: (_data, variables) => {
      removeInsight(variables.id);
    },
  });

  const isPinned = (msg: ChatMessage) => insights.find((i) => i.text === msg.content);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    addMessage({ role: "user", content: trimmed });
    scrollToBottom();
    setInput("");
    setSending(true);

    try {
      // Determine effective model identifier to send to backend
      const modelParam = provider === "local" ? modelId ?? undefined : provider;
      const resp = await postChat(trimmed, [...history, { role: "user", content: trimmed }], modelParam);
      addMessage({ role: "assistant", content: resp.answer, contexts: resp.contexts, citations: resp.citations });

      // If this was the first exchange, finalize conversation now
      if (!activeId) {
        const title = generateTitle(trimmed);
        finalizeConversation(title);
      }
    } catch (err) {
      addMessage({ role: "assistant", content: `Error: ${String(err)}` });
    } finally {
      setSending(false);
      scrollToBottom();
    }
  };

  // Helper: convert Markdown → safe HTML
  const toHtml = (md: string): string => {
    try {
      const htmlOrPromise = marked.parse(md);
      if (typeof htmlOrPromise === "string") {
        return DOMPurify.sanitize(htmlOrPromise);
      }
      // Fallback for async parse returning Promise – not awaited during render
      return "";
    } catch {
      return md;
    }
  };

  // Helper to generate succinct title from first user query
  const generateTitle = (text: string): string => {
    // Remove markdown and condense whitespace
    const clean = text.replace(/[\#*_`\[\]()]/g, "").replace(/\s+/g, " ").trim();
    const words = clean.split(" ");
    const stop = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","can","this","that","these","those"]);
    const meaningful: string[] = [];
    for (const w of words) {
      const wc = w.toLowerCase().replace(/[^\w]/g, "");
      if (wc.length > 2 && !stop.has(wc) && !/^\d+$/.test(wc)) {
        meaningful.push(w.charAt(0).toUpperCase() + w.slice(1));
        if (meaningful.length >= 4) break;
      }
    }
    const title = meaningful.length ? meaningful.slice(0,4).join(" ") : clean.slice(0,20);
    return title;
  };

  // Mascot component with wiggle animation
  const Mascot: React.FC = () => {
    const [wiggle, setWiggle] = useState(false);
    return (
      <img
        src={squancePng}
        alt="Agent"
        className={clsx("w-10 h-10 mt-1 cursor-pointer select-none", wiggle && "animate-wiggle")}
        onClick={() => setWiggle(true)}
        onAnimationEnd={() => setWiggle(false)}
      />
    );
  };

  const [editingIdx, setEditingIdx] = useState<number|null>(null);
  const [editValue, setEditValue] = useState("");

  // Index of the message currently awaiting delete confirmation
  const [deleteConfirmIdx, setDeleteConfirmIdx] = useState<number | null>(null);

  return (
    <div className="flex flex-col h-full max-h-screen relative p-4 pt-0">
      <div className="flex-1 overflow-y-auto space-y-4 p-4 pb-28 chat-scroll" ref={messageContainerRef} onMouseUp={handleSelection} onKeyUp={handleSelection}>
        {history.map((msg: ChatMessage, idx: number) => {
          const muted = msg.muted;
          const bubbleClasses = muted ? "inline-block bg-chat-assistant-bg text-gray-400 italic rounded-full px-4 py-1 max-w-xl" :
            msg.role === "user"
              ? "inline-block bg-blue-600 text-white rounded-lg px-4 py-2 max-w-xl shadow-lg "
              : "block bg-chat-assistant-bg text-defaultText rounded-lg px-4 py-2 w-full relative assistant-bubble shadow-inner";

          const contentHtml = toHtml(msg.content);

          return (
            <div key={idx} className={msg.role === "user" ? "text-right" : "text-left"}>
              <div className={bubbleClasses}>
                {muted ? (
                  <span className="text-xs">(Message hidden)</span>
                ) : editingIdx === idx ? (
                  <textarea
                    className="w-full h-32 bg-primaryBg text-defaultText border border-primaryBg rounded p-2"
                    value={editValue}
                    onChange={(e)=>setEditValue(e.target.value)}
                    autoFocus
                  />
                ) : (
                  <div dangerouslySetInnerHTML={{ __html: contentHtml }} />
                )}

                {msg.role === "assistant" && !muted && (
                  <button
                    className="absolute -top-2 -right-2 w-7 h-7 p-0 flex items-center justify-center rounded-full bg-transparent border-0 hover:bg-white/10 focus:outline-none"
                    title={isPinned(msg) ? "Unpin insight" : "Pin insight"}
                    onClick={() => {
                      const existing = isPinned(msg);
                      if (existing) {
                        deleteMutation.mutate({ id: existing.id });
                      } else {
                        addMutation.mutate(msg);
                      }
                    }}
                  >
                    {isPinned(msg) ? (
                      <PinRedIcon className="w-5 h-5 pointer-events-none" />
                    ) : (
                      <PinIcon className="w-5 h-5 pointer-events-none opacity-60" />
                    )}
                  </button>
                )}

                { !muted && msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                  <details className="mt-2 text-xs text-gray-600">
                    <summary className="cursor-pointer select-none">Sources</summary>
                    <ul className="list-decimal ml-4 space-y-1">
                      {msg.citations.map((c: Record<string, unknown>, i: number) => {
                        const title = (c as any).title as string | undefined;
                        const source = (c as any).source as string | undefined;
                        const year = (c as any).year as string | undefined;
                        const section = (c as any).section as string | undefined;
                        const page = (c as any).page as number | undefined;
                        let display = title ? `${title} (${source ?? ""}${year ? ", " + year : ""})` : source ?? "";
                        if (section) {
                          display += ` – ${section}`;
                        }
                        if (page) {
                          display += `, p.${page}`;
                        }
                        const previewTxt = (msg.contexts && msg.contexts[i]) ? msg.contexts[i] : "";
                        return (
                          <li key={i}>
                            {display}
                            <button
                              onClick={() => source && setPdfViewer({ filename: source, page })}
                              title={previewTxt}
                              className="inline-flex w-4 h-4 ml-1 p-0 align-baseline items-center justify-center translate-y-1"
                            >
                              <MagnifyIcon className="w-4 h-4" />
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </details>
                )}

                
              </div>
              {/* Action row & mascot */}
              <div className={msg.role === "assistant" ? "flex items-top gap-1 mt-1" : "flex justify-end gap-1 mt-1"}>
                {msg.role === "assistant" && !muted && <Mascot />}

                {/* Pin bottom (white icon) for assistant */}
                {msg.role === "assistant" && !muted && (
                  <button
                    className="w-5 h-5 p-0 mt-1 flex items-center justify-center bg-transparent border-0 text-defaultText hover:bg-white/10 focus:outline-none"
                    title={isPinned(msg) ? "Unpin insight" : "Pin insight"}
                    onClick={() => {
                      const existing = isPinned(msg);
                      if (existing) {
                        deleteMutation.mutate({ id: existing.id });
                      } else {
                        addMutation.mutate(msg);
                      }
                    }}
                  >
                    {isPinned(msg) ? <PinRedIcon className="w-4 h-4" /> : <PinIcon className="w-4 h-4" />}
                  </button>
                )}

                {/* Edit */}
                {!muted && (
                  <button className="w-5 h-5 p-0 mt-1 flex items-center justify-center bg-transparent border-0 text-defaultText hover:bg-white/10 focus:outline-none" title="Edit message" onClick={() => {
                    setEditingIdx(idx);
                    setEditValue(msg.content);
                  }}>
                    <PencilIcon className="w-4 h-4" />
                  </button>
                )}

                {/* Hide/Show */}
                <button className="w-5 h-5 p-0 mt-1 flex items-center justify-center bg-transparent border-0 text-defaultText hover:bg-white/10 focus:outline-none" title={msg.muted ? "Show message" : "Hide message"} onClick={() => {
                  updateMessage(idx, (m) => ({ ...m, muted: !m.muted }));
                }}>
                  {msg.muted ? <EyeOpenIcon className="w-4 h-4" /> : <EyeClosedIcon className="w-4 h-4" />}
                </button>

                {/* Delete / Confirm */}
                {deleteConfirmIdx === idx ? (
                  <>
                    {/* Confirm delete button */}
                    <button
                      className="flex px-2 py-1 h-5 items-center justify-center text-xs rounded mt-1 bg-[#db363c]/50 hover:bg-[#db363c] focus:outline-none"
                      title="Delete message"
                      onClick={() => {
                        deleteMessage(idx);
                        setDeleteConfirmIdx(null);
                      }}
                    >
                      Delete
                    </button>
                    {/* Cancel button */}
                    <button
                      className="w-5 h-5 p-0 mt-1 flex items-center justify-center bg-transparent border-0 text-defaultText hover:bg-white/10 focus:outline-none"
                      title="Cancel"
                      onClick={() => setDeleteConfirmIdx(null)}
                    >
                      <CloseIcon className="w-4 h-4 pointer-events-none" />
                    </button>
                  </>
                ) : (
                  !muted && (
                    <button
                      className="w-5 h-5 p-0 mt-1 flex items-center justify-center bg-transparent border-0 text-defaultText hover:bg-white/10 focus:outline-none"
                      title="Delete message"
                      onClick={() => setDeleteConfirmIdx(idx)}
                    >
                      <TrashIcon className="w-4 h-4 pointer-events-none" />
                    </button>
                  )
                )}
              </div>
            </div>
          );
        })}
        {isSending && (
          <div className="text-left">
            <div className="inline-block bg-gray-200 text-gray-900 rounded-lg px-4 py-2 max-w-xl animate-pulse">
              Generating...
            </div>
          </div>
        )}
        {pdfViewer && (
          <PdfModal
            filename={pdfViewer.filename}
            initialPage={pdfViewer.page}
            onClose={() => setPdfViewer(null)}
          />
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="absolute bottom-0 left-0 w-full p-4 flex gap-2 items-center backdrop-blur-lg bg-primaryBg/60">
        {/* Context pool selector dropdown */}
        <div className="relative">
          <button
            className="bg-buttonBg text-defaultText p-2 rounded disabled:opacity-50 flex items-center justify-center"
            onClick={() => setCtxDropdownOpen((prev) => !prev)}
            disabled={isSending}
            title="Select context pool"
          >
            <BookIcon className="w-5 h-5" />
          </button>
          {ctxDropdownOpen && (
            <div
              ref={dropdownRef}
              className="absolute bottom-12 left-0 bg-primaryBg bg-opacity-80 backdrop-blur-lg text-defaultText border border-primaryBg/80 rounded shadow-lg p-3 space-y-2 z-50 min-w-[10rem] w-max max-w-[20rem]"
            >
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="ctxpool"
                  value="all"
                  checked={selectedCollectionId === null}
                  onChange={(e) => handleCtxPoolChange(e.target.value)}
                />
                All papers
              </label>
              {collections?.map((c) => (
                <label key={c.id} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="ctxpool"
                    value={c.id}
                    checked={selectedCollectionId === c.id}
                    onChange={(e) => handleCtxPoolChange(e.target.value)}
                  />
                  {c.name}
                </label>
              ))}

              {/* Create new collection option */}
              <button
                className="flex items-center gap-2 w-full mt-2 px-2 py-1 rounded-full bg-buttonBg/60 hover:bg-white/10 focus:outline-none"
                onClick={() => {
                  setCtxDropdownOpen(false);
                  setActiveTab("Library");
                  requestFocusNewCollection();
                }}
              >
                <PlusIcon className="w-4 h-4" />
                <span>Create new collection</span>
              </button>
            </div>
          )}
        </div>

        <input
          className="flex-1 border rounded px-3 py-2 focus:outline-none bg-primaryBg bg-opacity-60 backdrop-blur-lg text-defaultText border-primaryBg"
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={isSending}
        />

        <button
          className="bg-buttonBg text-defaultText px-4 py-2 rounded disabled:opacity-50"
          onClick={handleSend}
          disabled={isSending || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
} 