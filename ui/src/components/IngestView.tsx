import { useState, useEffect } from "react";
import { ingestPdfs, getTaskStatus } from "../api";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { AxiosError } from "axios";
import FolderIcon from "../assets/folder-yellow.svg?react";
import type { DragEvent as ReactDragEvent } from "react";
import { useToast } from "./ToastProvider";

export default function IngestView() {
  const [pdfDir, setPdfDir] = useState("data/papers");
  const [taskId, setTaskId] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // Drag-and-drop import of external files
  // ---------------------------------------------------------------------------

  // Highlight state when user is dragging files over the ingest area
  const [isDragActive, setIsDragActive] = useState(false);
  const toast = useToast();

  const handleDragOver = (e: ReactDragEvent<HTMLDivElement>) => {
    // Accept only file drags – ignore others (e.g., text selections)
    if (Array.from(e.dataTransfer.types).includes("Files")) {
      e.preventDefault(); // Allow drop
      e.dataTransfer.dropEffect = "copy";
    }
  };

  const ALLOWED_EXTS = ["pdf", "txt", "md", "doc", "docx"];

  const handleDrop = async (e: ReactDragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    setIsDragActive(false);

    const files = Array.from(e.dataTransfer.files ?? []);
    if (!files.length) return;

    try {
      // Ensure repository directory exists. We use the same behaviour as the
      // root-level handler: create relative to the app data directory so that
      // "data/papers" resolves to the expected location on disk.
      const { mkdir, writeFile, BaseDirectory } = await import("@tauri-apps/plugin-fs");

      // Strip a leading "data/" prefix because we map the *Data* base dir to
      // the root of our workspace. This keeps the on-disk path consistent with
      // the server-side default (data/papers).
      const repoDir = pdfDir.startsWith("data/") ? pdfDir.replace(/^data\//, "") : pdfDir;

      await mkdir(repoDir, { recursive: true, baseDir: BaseDirectory.Data });

      const copied: string[] = [];

      for (const file of files) {
        const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
        if (!ALLOWED_EXTS.includes(ext)) {
          // Skip unsupported types – inform the user but continue with others.
          toast(`Unsupported file type: ${file.name}`, "info");
          continue;
        }

        const buf = await file.arrayBuffer();
        await writeFile(`${repoDir}/${file.name}`, new Uint8Array(buf), {
          baseDir: BaseDirectory.Data,
        });
        copied.push(file.name);
      }

      if (copied.length) {
        toast(`📄 Added ${copied.length} file(s) to repository`, "success");
      }
    } catch (err) {
      toast(`Failed to import file(s): ${String(err)}`, "error");
      throw err; // Fail fast per project policy
    }
  };

  // Visual feedback for drag-over state
  const onDragEnter = (e: ReactDragEvent<HTMLDivElement>) => {
    if (Array.from(e.dataTransfer.types).includes("Files")) {
      setIsDragActive(true);
    }
  };

  const onDragLeave = (e: ReactDragEvent<HTMLDivElement>) => {
    // Only reset when leaving the component bounds (relatedTarget null)
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragActive(false);
    }
  };

  const ingestMutation = useMutation<string, Error, string>({
    mutationFn: ingestPdfs,
    onSuccess: (id: string) => {
      setTaskId(id);
    },
  });

  // Poll background task progress
  const { data: taskData, error: taskErr, isError } = useQuery({
    queryKey: ["task-status", taskId],
    queryFn: () => (taskId ? getTaskStatus(taskId) : Promise.resolve(null)),
    enabled: !!taskId,
    refetchInterval: 3000,
    retry: false,
  });

  useEffect(() => {
    if (isError) {
      const axErr = taskErr as AxiosError | undefined;
      const status = axErr?.response?.status;
      if (status === 404) {
        console.warn("Background task not found on server – stopping polling.");
        setTaskId(null);
      }
    }
  }, [isError, taskErr]);

  const handleOpenFolder = async () => {
    try {
      const w = window as any;
      const isTauri =
        Boolean(w.__TAURI__) ||
        Boolean(w.__TAURI_INTERNALS__) ||
        Boolean(w.isTauri) ||
        navigator.userAgent.includes("Tauri");
      if (isTauri) {
        const { dataDir, join } = await import("@tauri-apps/api/path");

        // Remove a leading "data/" so that "data/papers" maps to <dataDir>/papers
        const relative = pdfDir.startsWith("data/") ? pdfDir.replace(/^data\//, "") : pdfDir;

        const base = await dataDir();
        const abs = await join(base, relative);

        // Ensure the directory exists using a direct plugin invoke to avoid the browser stub during dev-server builds.
        const { invoke } = await import("@tauri-apps/api/core");
        const { BaseDirectory } = await import("@tauri-apps/api/path");

        await invoke("plugin:fs|mkdir", {
          path: relative,
          options: { recursive: true, baseDir: BaseDirectory.Data },
        });

        // @ts-ignore – plugin types not shipped to the web build.
        const { openPath } = await import("@tauri-apps/plugin-opener");
        // Open the directory in the OS file-manager (Explorer / Finder).
        await openPath(abs);
      } else {
        // For web preview fall back to raw file:// URL (may be blocked by browser)
        window.open(`file://${pdfDir}`, "_blank");
      }
    } catch (err) {
      console.error(err);
      toast(`Failed to open folder: ${String(err)}`, "error");
    }
  };

  return (
    <div
      className={
        "p-6 space-y-4 max-w-xl mx-auto border-2 rounded-md transition-colors " +
        (isDragActive ? "border-accentText bg-secondaryBg/40" : "border-transparent")
      }
      onDragEnter={onDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={onDragLeave}
      onDrop={handleDrop}
    >
      <h1 className="text-xl font-semibold">Data Ingestion</h1>
      <div className="space-y-2">
        <label className="text-sm font-medium">PDF Directory</label>
        <input
          className="border rounded px-2 py-1 w-full text-sm bg-headerBg text-light-text border-primaryBg"
          value={pdfDir}
          onChange={(e) => setPdfDir(e.target.value)}
        />
      </div>
      <div className="flex gap-2">
        <button
          className="flex-1 bg-buttonBg text-defaultText rounded px-3 py-1 disabled:opacity-50"
          onClick={() => ingestMutation.mutate(pdfDir)}
          disabled={ingestMutation.isPending}
        >
          Ingest PDFs
        </button>
        <button
          className="px-3 py-1 rounded"
          onClick={handleOpenFolder}
          title="Open folder"
        >
          <FolderIcon className="w-5 h-5 flex-shrink-0" />
        </button>
      </div>
      {taskId && (
        <div className="text-sm">
          {taskData && (taskData as any).status === "done" && (
            <span className="text-green-600">Ingestion complete!</span>
          )}
          {taskData && (taskData as any).status === "error" && (
            <span className="text-red-600">Ingestion failed: {(taskData as any).error}</span>
          )}
          {(!taskData || (taskData as any).status === "running" || (taskData as any).status === "pending") && (
            <span className="text-yellow-600">⏳ Ingestion running…</span>
          )}
        </div>
      )}
    </div>
  );
} 