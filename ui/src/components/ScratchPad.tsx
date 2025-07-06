import { useState, Suspense, useEffect, useRef, useMemo, useCallback } from "react";
import { useDroppable, useDndMonitor } from "@dnd-kit/core";
import { createNote, deleteNote } from "../api";
import { useMutation } from "@tanstack/react-query";
import DOMPurify from "dompurify";
import "react-quill/dist/quill.snow.css";
import NotepadModal from "./NotepadModal";
import PinIcon from "../assets/pin-red.svg?react";
import FolderIcon from "../assets/folder-yellow.svg?react";
import PlusIcon from "../assets/plus.svg?react";

// Dynamically import ReactQuill via React.lazy so the bundle loads only when needed.
import { lazy } from "react";
// Ensure our custom blot is registered before the editor mounts
import "../quill/InsightBlot";

const QuillEditor = lazy(() => import("react-quill"));

// @ts-ignore – mark.js has no bundled types
import Mark from "mark.js";

export default function ScratchPad() {
  const [title, setTitle] = useState("Untitled Note");
  const [content, setContent] = useState("<p></p>");
  const [markdown, setMarkdown] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [highlightTerm, setHighlightTerm] = useState("");
  const containerRef = useRef<HTMLDivElement | null>(null);
  const quillRef = useRef<any>(null); // react-quill instance
  const markInstance = useRef<any>(null);
  const saveBtnRef = useRef<HTMLButtonElement | null>(null);
  const [noteId, setNoteId] = useState<number | null>(null);

  // ──────────────────────────────── Persistence & auto-save ──
  const LOCAL_KEY = "scratchpad-state";
  const autoSaveTriggered = useRef(false);
  const createdInSessionRef = useRef(false);

  // Restore unsaved state on mount
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(LOCAL_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          setTitle(parsed.title ?? "Untitled Note");
          setContent(parsed.content ?? "<p></p>");
          setNoteId(parsed.noteId ?? null);
          createdInSessionRef.current = Boolean(parsed.createdInSession);
        }
      }
    } catch {
      /* ignore corrupted sessionStorage */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist current editing state for the lifetime of the tab
  useEffect(() => {
    try {
      sessionStorage.setItem(
        LOCAL_KEY,
        JSON.stringify({ title, content, noteId, createdInSession: createdInSessionRef.current })
      );
    } catch {
      /* ignore quota errors */
    }
  }, [title, content, noteId]);

  // Initialize a real saved note when the user starts typing.
  useEffect(() => {
    if (noteId !== null || autoSaveTriggered.current) return;
    const plainText = content.replace(/<[^>]+>/g, "").trim();
    if (plainText.length === 0) return;

    autoSaveTriggered.current = true;
    exportMut.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [content, noteId]);

  useEffect(() => {
    if (containerRef.current) {
      markInstance.current = new Mark(containerRef.current);
    }
    return () => {
      markInstance.current = null;
    };
  }, []);

  useEffect(() => {
    if (!markInstance.current) return;
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

  const handleSelection = () => {
    const sel = window.getSelection();
    if (!sel) {
      setHighlightTerm("");
      return;
    }
    const text = sel.toString().trim();
    if (text.length > 0 && text.length <= 60) {
      setHighlightTerm(text);
    } else {
      setHighlightTerm("");
    }
  };

  const exportMut = useMutation<number | null, Error, void>({
    mutationFn: async () => {
      // Convert HTML to Markdown if turndown is available
      const safeHtml = DOMPurify.sanitize(content);
      let md = safeHtml;
      try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/ban-ts-comment
        // @ts-ignore
        const TurndownService = require("turndown").default;
        // eslint-disable-next-line @typescript-eslint/no-unsafe-call
        const td = new TurndownService();
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
        md = td.turndown(safeHtml);
      } catch {
        // keep html
      }
      setMarkdown(md);
      // Determine the appropriate title, adding a timestamp if still default.
      let finalTitle = title.trim() || "Untitled Note";
      if (finalTitle === "Untitled Note") {
        finalTitle = `${finalTitle} ${new Date().toLocaleString()}`;
        setTitle(finalTitle);
      }

      if (noteId !== null) {
        // Existing note – skip duplicate creation. Return current id.
        return noteId;
      }
      // Mark that this note will be created in this session.
      createdInSessionRef.current = true;
      return createNote(md, finalTitle);
    },
    onSuccess: (pid) => {
      flashSaveButton();
      setMarkdown("");
      if (noteId === null && pid !== null) {
        setNoteId(pid);
      }
    },
    onError: (err) => alert(String(err)),
  });

  // ───────────────────────────────────────── Quill configuration ──
  const quillModules = useMemo(
    () => ({
      toolbar: [
        [{ header: [1, 2, 3, false] }],
        ["bold", "italic", "underline", "strike"],
        [{ list: "ordered" }, { list: "bullet" }],
        [{ color: [] }, { background: [] }],
        [{ align: [] }],
        ["blockquote", "code-block"],
        ["link", "image", "video"],
        ["clean"],
      ],
    }),
    []
  );

  const quillFormats = [
    "header",
    "bold",
    "italic",
    "underline",
    "strike",
    "list",
    "color",
    "background",
    "align",
    "blockquote",
    "code-block",
    "link",
    "image",
    "video",
    "insight", // our custom blot
  ];

  // ───────────────────────────────────────── dnd-kit drop handling ──
  const DROPPABLE_ID = "scratchpad-editor";
  const { setNodeRef: setDropRef } = useDroppable({ id: DROPPABLE_ID });

  useDndMonitor({
    onDragEnd(event: any) {
      try {
        if (event.over?.id !== DROPPABLE_ID) return;
        const data = event.active.data?.current as any;
        if (!data || data.type !== "insight") return;

        const quill = quillRef.current?.getEditor?.();
        if (!quill) throw new Error("Quill editor not ready.");

        const pos = quill.getSelection()?.index ?? quill.getLength();
        quill.insertEmbed(pos, "insight", { title: data.title ?? "", body: data.body ?? "" }, "user");
        quill.insertText(pos + 1, "\n", "user");
      } catch (err) {
        // eslint-disable-next-line no-alert
        alert(String(err));
        throw err;
      }
    },
  });

  const combinedContainerRef = useCallback((node: HTMLDivElement | null) => {
    containerRef.current = node;
    setDropRef(node);
  }, [setDropRef]);

  const flashSaveButton = () => {
    const btn = saveBtnRef.current;
    if (!btn) return;
    let count = 0;
    const id = setInterval(() => {
      btn.style.filter = btn.style.filter ? "" : "brightness(2)";
      count++;
      if (count >= 4) {
        btn.style.filter = "";
        clearInterval(id);
      }
    }, 150);
  };

  // ───────────────────────────────────────── Make toolbar sticky ──
  useEffect(() => {
    const quill = quillRef.current?.getEditor?.();
    if (!quill) return;
    try {
      const toolbar: HTMLElement | undefined = (quill.getModule("toolbar") as any)?.container;
      if (toolbar) {
        // Apply frosted-glass appearance and stickiness (only once)
        if (!toolbar.dataset.frosted) {
          toolbar.classList.add(
            "top-0", "z-20",
            "backdrop-blur-lg", // blur background behind the bar
            "backdrop-saturate-150", // boost saturation a little
            "text-defaultText" // use theme's default text color
          );

          // Compute parent background colour to inherit theme and add transparency
          const parentBg = getComputedStyle(toolbar.parentElement ?? toolbar).backgroundColor;
          // parentBg may be 'rgb(r, g, b)' or 'rgba(r, g, b, a)'
          const rgbMatch = parentBg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
          if (rgbMatch) {
            const [_, r, g, b] = rgbMatch;
            toolbar.style.backgroundColor = `rgba(${r}, ${g}, ${b}, 0.6)`;
          } else {
            // Fallback to CSS variable-based colour (primary background) with 0.6 alpha
            toolbar.style.backgroundColor = "rgb(from var(--color-secondary-bg) r g b / 0.6)";
          }
          toolbar.style.backdropFilter = "blur(16px) saturate(150%)";
          // Safari/WebKit prefix (WebView2 uses Chromium so not strictly needed, but harmless)
          (toolbar.style as any).webkitBackdropFilter = "blur(16px) saturate(150%)";

          toolbar.style.position = "sticky";
          toolbar.style.top = "0";
          toolbar.style.zIndex = "20";

          toolbar.dataset.frosted = "true";
        }

        // Assign descriptive titles to each toolbar button (do once)
        if (!toolbar.dataset.titled) {
          const tooltipMap: Record<string, string> = {
            bold: "Bold (Ctrl+B)",
            italic: "Italic (Ctrl+I)",
            underline: "Underline (Ctrl+U)",
            strike: "Strikethrough",
            "header-value-1": "Heading 1",
            "header-value-2": "Heading 2",
            "header-value-3": "Heading 3",
            "list-value-ordered": "Numbered List",
            "list-value-bullet": "Bulleted List",
            color: "Text Color",
            background: "Background Color",
            align: "Text Alignment",
            blockquote: "Blockquote",
            "code-block": "Code Block",
            link: "Insert Link",
            image: "Insert Image",
            video: "Insert Video",
            clean: "Remove Formatting",
          };

          toolbar.querySelectorAll("button").forEach((btn) => {
            const quillClass = Array.from(btn.classList).find((c) => c.startsWith("ql-"));
            if (!quillClass) return;
            let key = quillClass.replace("ql-", "");
            // Handle buttons dependent on value attributes, like header and list.
            if (key === "header" || key === "list" || key === "align") {
              const val = btn.getAttribute("value") ?? "";
              key = `${key}-value-${val}`;
            }
            const title = (tooltipMap as Record<string, string>)[key];
            if (title) {
              (btn as HTMLButtonElement).title = title;
            }
          });
          toolbar.dataset.titled = "true";
        }
      }
    } catch {
      /* ignore */
    }
  }, []);

  // ──────────────────────────────── Auto-delete Untitled empty note ──
  useEffect(() => {
    if (noteId === null) return;
    if (!createdInSessionRef.current) return;
    if (!title.includes("Untitled Note")) return;
    // Strip HTML to detect emptiness
    const plainText = content.replace(/<[^>]+>/g, "").trim();
    if (plainText.length > 0) return;

    // Delete and reset
    (async () => {
      try {
        await deleteNote(noteId);
      } catch (err) {
        // eslint-disable-next-line no-alert
        alert(String(err));
      } finally {
        setTitle("Untitled Note");
        setContent("<p></p>");
        setMarkdown("");
        setNoteId(null);
        autoSaveTriggered.current = false;
        createdInSessionRef.current = false;
        sessionStorage.removeItem(LOCAL_KEY);
      }
    })();
  }, [content, title, noteId]);

  // ───────────────────────────────────────── Auto-focus on mount ──
  useEffect(() => {
    // Attempt to focus the Quill editor once it's available
    const timer = setInterval(() => {
      const quill = quillRef.current?.getEditor?.();
      if (quill) {
        quill.focus();
        clearInterval(timer);
      }
    }, 100);
    return () => clearInterval(timer);
  }, []);

  return (
    <div
      className="m-6 p-8 flex flex-col gap-4 h-full rounded-lg bg-secondaryBg shadow-inner"
      ref={combinedContainerRef}
      onMouseUp={handleSelection}
      onKeyUp={handleSelection}
    >
      <NotepadModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        onSelect={(id, noteTitle, html) => {
          setNoteId(id);
          setTitle(noteTitle);
          setContent(html);
        }}
      />
      <div className="flex items-center gap-3">
        <input
          className="text-lg font-semibold flex-1 bg-transparent border-b border-defaultText/80 focus:outline-none"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <button
          title="Export note"
          className="disabled:opacity-50 p-0 w-8 h-8 inline-flex items-center justify-center"
          onClick={() => exportMut.mutate()}
          disabled={exportMut.isPending}
          ref={saveBtnRef}
        >
          <PinIcon className="w-5 h-5 flex-shrink-0" />
        </button>
        <button
          title="Open notepads"
          className="p-0 w-8 h-8 inline-flex items-center justify-center"
          onClick={() => setShowModal(true)}
        >
          <FolderIcon className="w-5 h-5 flex-shrink-0" />
        </button>
        <button
          title="New note"
          className="p-0 w-8 h-8 inline-flex items-center justify-center"
          onClick={() => {
            setTitle("Untitled Note");
            setContent("<p></p>");
            setMarkdown("");
            setNoteId(null);
            autoSaveTriggered.current = false;
            sessionStorage.removeItem(LOCAL_KEY);
            createdInSessionRef.current = false;
          }}
        >
          <PlusIcon className="w-5 h-5 flex-shrink-0 text-defaultText" />
        </button>
      </div>
      <Suspense
        fallback={
          <textarea
            className="w-full flex-1 border rounded p-2"
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />
        }
      >
        <div className="flex-1 flex flex-col overflow-hidden">
          <QuillEditor
            ref={quillRef}
            theme="snow"
            value={content}
            onChange={setContent}
            modules={quillModules}
            formats={quillFormats}
            className="scratchpad-quill flex-1 flex flex-col"
          />
        </div>
      </Suspense>
      {markdown && !exportMut.isPending && (
        <pre className="bg-gray-100 p-2 text-xs overflow-x-auto whitespace-pre-wrap break-words">
          {markdown}
        </pre>
      )}
    </div>
  );
} 