// (React import removed – not required for React 17+ JSX transform)
import { listPapers, createNote, deleteNote } from "../api";
import { marked } from "marked";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import CloseIcon from "../assets/close.svg?react";
import PlusIcon from "../assets/plus.svg?react";
import TrashIcon from "../assets/trash.svg?react";
import { useState } from "react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (id: number, title: string, html: string) => void;
}

/**
 * NotepadModal displays a list of existing Markdown/TXT notes that were previously
 * exported via the ScratchPad. Users can pick one to load it back into the editor, or
 * add a new external file (markdown/txt/docx) via the quick-access button at the bottom.
 */
export default function NotepadModal({ isOpen, onClose, onSelect }: Props) {
  const queryClient = useQueryClient();

  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null);

  // Fetch all papers and keep only markdown/txt/docx files that live in the `notes/` dir.
  const { data: papers } = useQuery({ queryKey: ["papers"], queryFn: listPapers });

  const filtered = (papers ?? []).filter((p) => /\.(md|markdown|txt|docx)$/i.test(p.filename));

  // Mutation to import an external file and create a new note record on the server.
  const importMut = useMutation<number, Error, void>({
    mutationFn: async () => {
      // Show open-file dialog
      const selected = await openDialog({
        filters: [
          { name: "Notes", extensions: ["md", "markdown", "txt", "docx"] },
        ],
        multiple: false,
      });

      if (typeof selected !== "string") return Promise.reject(new Error("No file selected"));

      // Handle txt/md directly; docx requires conversion.
      const lower = selected.toLowerCase();
      let mdContent = "";
      if (lower.endsWith(".md") || lower.endsWith(".markdown") || lower.endsWith(".txt")) {
        mdContent = await readTextFile(selected);
      } else if (lower.endsWith(".docx")) {
        // @ts-ignore – mammoth has no types bundled
        const mammoth: any = await import("mammoth");
        const { value } = await mammoth.convertToMarkdown({ path: selected });
        mdContent = value;
      } else {
        return Promise.reject(new Error(`Unsupported file type: ${selected}`));
      }

      // Derive title from first markdown heading or filename fallback.
      const firstLine = mdContent.split(/\r?\n/).find((l) => l.trim());
      const title = firstLine ? firstLine.replace(/^#+\s*/, "") : selected.split(/[/\\]/).pop() ?? "Imported note";

      return createNote(mdContent, title);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["papers"] });
    },
  });

  const deleteMutation = useMutation<void, Error, number>({
    mutationFn: deleteNote,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["papers"] });
    },
  });

  // Early return if modal not open – avoids DOM overhead.
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-lg w-96 max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()} // prevent background click closing when interacting inside
      >
        <header className="p-4 border-b border-gray-300 dark:border-gray-700 flex justify-between items-center">
          <h3 className="font-semibold">Notepads</h3>
          <button onClick={onClose} className="w-5 h-5 p-0 flex items-center justify-center" aria-label="Close">
            <CloseIcon className="w-5 h-5 flex-shrink-0" />
          </button>
        </header>
        <ul className="flex-1 overflow-y-auto divide-y divide-gray-200 dark:divide-gray-700">
          {filtered.length > 0 ? (
            filtered.map((p) => (
              <li key={p.id}>
                {deleteConfirmId === p.id ? (
                  <div className="flex items-center justify-end gap-1 px-4 py-2">
                    <button
                      className="flex px-2 py-1 h-5 items-center justify-center text-xs rounded bg-[#db363c]/50 hover:bg-[#db363c] focus:outline-none"
                      onClick={() => deleteMutation.mutate(p.id)}
                    >
                      Delete
                    </button>
                    <button
                      className="w-5 h-5 p-0 flex items-center justify-center bg-transparent border-0 hover:bg-gray-400/20 focus:outline-none rounded"
                      onClick={() => setDeleteConfirmId(null)}
                    >
                      <CloseIcon className="w-4 h-4 pointer-events-none" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer" onClick={async ()=>{
                    try {
                      const md = await readTextFile(`notes/${p.filename}`);
                      const htmlOrPromise = marked.parse(md);
                      const html = typeof htmlOrPromise === "string" ? htmlOrPromise : await htmlOrPromise;
                      onSelect(p.id, p.title ?? p.filename, html);
                      onClose();
                    } catch(err){
                      alert(`Failed to load note: ${String(err)}`);
                    }
                  }}>
                    <span className="flex-1 truncate">{p.title ?? p.filename}</span>
                    <button
                      className="w-4 h-4 ml-2 p-0 flex items-center justify-center opacity-60 hover:opacity-100 bg-transparent border-0"
                      title="Delete note"
                      onClick={(e)=>{e.stopPropagation(); setDeleteConfirmId(p.id);}}
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </li>
            ))
          ) : (
            <li className="p-4 text-sm text-gray-500">No notepads found.</li>
          )}
        </ul>
        <footer className="p-4 border-t border-gray-300 dark:border-gray-700 flex items-center gap-2">
          <button
            className="flex-1 bg-blue-600 text-white px-3 py-1 rounded disabled:opacity-50"
            onClick={() => importMut.mutate()}
            disabled={importMut.isPending}
          >
            <PlusIcon className="w-4 h-4" /> Add from file
          </button>
        </footer>
      </div>
    </div>
  );
} 