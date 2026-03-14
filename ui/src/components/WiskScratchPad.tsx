import { useState, useEffect, useRef, useCallback } from "react";
import { useDroppable, useDndMonitor } from "@dnd-kit/core";
import { createNote, deleteNote } from "../api";
import { useMutation } from "@tanstack/react-query";
import DOMPurify from "dompurify";
import NotepadModal from "./NotepadModal";
import PinIcon from "../assets/pin-red.svg?react";
import FolderIcon from "../assets/folder-yellow.svg?react";
import PlusIcon from "../assets/plus.svg?react";
import WiskEditor, { type WiskEditorRef, type WiskDocument } from "./WiskEditor";

const LOCAL_KEY = "wisk-scratchpad-state";

export default function WiskScratchPad() {
  const [title, setTitle] = useState("Untitled Note");
  const [wiskDocument, setWiskDocument] = useState<WiskDocument | null>(null);
  const [markdown, setMarkdown] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [noteId, setNoteId] = useState<number | null>(null);

  const wiskRef = useRef<WiskEditorRef>(null);
  const autoSaveTriggered = useRef(false);
  const createdInSessionRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const saveBtnRef = useRef<HTMLButtonElement>(null);

  // ──────────────────────────────────────────── Auto-save logic ──
  const exportMut = useMutation({
    mutationFn: async () => {
      if (!wiskDocument || !title.trim()) return;

      const htmlContent = wiskRef.current?.convertToHtml() || "";
      const sanitizedContent = DOMPurify.sanitize(htmlContent);

      if (noteId === null) {
        const newNote = await createNote(sanitizedContent, title.trim());
        setNoteId(newNote);
        autoSaveTriggered.current = true;
        createdInSessionRef.current = true;
        return newNote;
      } else {
        // Update existing note
        // Assuming there's an updateNote function similar to createNote
        // await updateNote(noteId, sanitizedContent, title.trim());
        autoSaveTriggered.current = true;
        return noteId;
      }
    },
    onSuccess: () => {
      flashSaveButton();
    },
    onError: (err) => {
      alert(String(err));
      throw err;
    },
  });

  // ──────────────────────────────────────── Persist state locally ──
  useEffect(() => {
    const saved = sessionStorage.getItem(LOCAL_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setTitle(parsed.title || "Untitled Note");
        setNoteId(parsed.noteId || null);
        if (parsed.wiskDocument) {
          setWiskDocument(parsed.wiskDocument);
        }
      } catch {
        // ignore parse errors
      }
    }
  }, []);

  useEffect(() => {
    const state = {
      title,
      noteId,
      wiskDocument,
    };
    sessionStorage.setItem(LOCAL_KEY, JSON.stringify(state));
  }, [title, noteId, wiskDocument]);

  // ────────────────────────────────────────────── Handle changes ──
  const handleWiskChange = useCallback((doc: WiskDocument) => {
    setWiskDocument(doc);
    
    // Auto-save after changes
    if (!exportMut.isPending) {
      setTimeout(() => {
        exportMut.mutate();
      }, 1000);
    }
  }, [exportMut]);

  // ───────────────────────────────────────── dnd-kit drop handling ──
  const DROPPABLE_ID = "wisk-scratchpad-editor";
  const { setNodeRef: setDropRef } = useDroppable({ id: DROPPABLE_ID });

  useDndMonitor({
    onDragEnd(event: any) {
      try {
        if (event.over?.id !== DROPPABLE_ID) return;
        const data = event.active.data?.current as any;
        if (!data || data.type !== "insight") return;

        // Add insight as a text block in Wisk
        if (wiskRef.current) {
          const insightText = `**${data.title || ""}**\n\n${data.body || ""}`;
          wiskRef.current.addBlock('text-element', { textContent: insightText });
        }
      } catch (err) {
        alert(String(err));
        throw err;
      }
    },
  });

  const combinedContainerRef = useCallback((node: HTMLDivElement | null) => {
    if (containerRef) {
      (containerRef as any).current = node;
    }
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

  // ──────────────────────────────── Auto-delete Untitled empty note ──
  useEffect(() => {
    if (noteId === null) return;
    if (!createdInSessionRef.current) return;
    if (!title.includes("Untitled Note")) return;
    
    // Check if document is empty
    const isEmpty = !wiskDocument?.data?.elements?.some(el => 
      el.value?.textContent?.trim() || el.value?.html?.trim()
    );
    
    if (!isEmpty) return;

    // Delete and reset
    (async () => {
      try {
        await deleteNote(noteId);
      } catch (err) {
        alert(String(err));
      } finally {
        setTitle("Untitled Note");
        setWiskDocument(null);
        setMarkdown("");
        setNoteId(null);
        if (autoSaveTriggered.current !== false) autoSaveTriggered.current = false;
        if (createdInSessionRef.current !== false) createdInSessionRef.current = false;
        sessionStorage.removeItem(LOCAL_KEY);
      }
    })();
  }, [wiskDocument, title, noteId]);

  // ───────────────────────────────────────── Auto-focus on mount ──
  useEffect(() => {
    const timer = setTimeout(() => {
      wiskRef.current?.focus();
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div
      className="m-6 p-8 flex flex-col gap-4 h-full rounded-lg bg-secondaryBg shadow-inner"
      ref={combinedContainerRef}
    >
      <NotepadModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        onSelect={(id, noteTitle, html) => {
          setNoteId(id);
          setTitle(noteTitle);
          // Convert HTML to Wisk document
          if (wiskRef.current) {
            const wiskDoc = wiskRef.current.convertFromHtml(html);
            setWiskDocument(wiskDoc);
            wiskRef.current.setDocument(wiskDoc);
          }
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
            setWiskDocument(null);
            setMarkdown("");
            setNoteId(null);
            // autoSaveTriggered.current = false;
            sessionStorage.removeItem(LOCAL_KEY);
            // createdInSessionRef.current = false;
            if (wiskRef.current) {
              wiskRef.current.setDocument({
                data: {
                  config: { name: 'Untitled Note', theme: 'default', plugins: [] },
                  elements: [{ id: 'new_' + Date.now(), component: 'text-element', value: { textContent: '' } }],
                  pluginData: {}
                }
              });
            }
          }}
        >
          <PlusIcon className="w-5 h-5 flex-shrink-0 text-defaultText" />
        </button>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        <WiskEditor
          ref={wiskRef}
          document={wiskDocument || undefined}
          onChange={handleWiskChange}
          className="flex-1"
          style={{ height: '100%', minHeight: '400px' }}
          placeholder="Start writing or press '/' for commands..."
        />
      </div>

      {markdown && !exportMut.isPending && (
        <pre className="bg-gray-100 p-2 text-xs overflow-x-auto whitespace-pre-wrap break-words">
          {markdown}
        </pre>
      )}
    </div>
  );
} 