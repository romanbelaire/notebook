import { useState, useEffect } from "react";
import { ingestPdfs, getTaskStatus, checkPaperHash, clearDatabase } from "../api";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { AxiosError } from "axios";
import type { DragEvent as ReactDragEvent } from "react";
import { useToast } from "./ToastProvider";
import { invoke } from "@tauri-apps/api/core";
import MagnifyIcon from "../assets/magnify.svg?react";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function IngestView() {
  // The repository directory for PDFs. Default is resolved to the app's data
  // directory (e.g. <AppData>/papers) on mount; the user may override via the
  // input or Browse button.
  const [pdfDir, setPdfDir] = useState<string>("");
  const [taskId, setTaskId] = useState<string | null>(() => {
    // Initialize from localStorage to persist across tab switches
    return localStorage.getItem('ingestion-task-id');
  });

  const queryClient = useQueryClient();

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

  // Function to calculate SHA256 hash of a file
  const calculateSHA256 = async (file: File): Promise<string> => {
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return hashHex;
  };

  // Function to extract PDF from Firefox PDF document (HTML wrapper)
  const extractPdfFromFirefoxDocument = async (file: File): Promise<Uint8Array | null> => {
    try {
      const text = await file.text();
      
      // Check if it's a Firefox PDF document
      if (!text.includes('<!DOCTYPE html') || !text.includes('pdf.js')) {
        return null; // Not a Firefox PDF document
      }
      
      // Look for the PDF data URL in the HTML
      const pdfDataMatch = text.match(/data:application\/pdf;base64,([A-Za-z0-9+/=]+)/);
      if (!pdfDataMatch) {
        console.warn('Firefox PDF document detected but no PDF data found');
        return null;
      }
      
      // Decode the base64 PDF data
      const base64Data = pdfDataMatch[1];
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      return bytes;
    } catch (err) {
      console.error('Error extracting PDF from Firefox document:', err);
      return null;
    }
  };

  const handleDrop = async (e: ReactDragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    setIsDragActive(false);

    const files = Array.from(e.dataTransfer.files ?? []);
    if (!files.length) return;

    try {
      // Detect if we're running in Tauri or web mode
      const w = window as any;
      const isTauri =
        Boolean(w?.__TAURI__) ||
        Boolean(w?.__TAURI_INTERNALS__) ||
        Boolean(w?.isTauri) ||
        navigator.userAgent.includes("Tauri");

      const copied: string[] = [];
      const skipped: string[] = [];

      if (isTauri) {
        // Tauri mode: Use Tauri file system API
        const { mkdir, writeFile, BaseDirectory } = await import("@tauri-apps/plugin-fs");
        const { dataDir, join } = await import("@tauri-apps/api/path");
        const base = await dataDir();
        let repoDir = pdfDir;
        if (pdfDir.startsWith(base)) {
          repoDir = pdfDir.slice(base.length);
          if (repoDir.startsWith("/") || repoDir.startsWith("\\")) {
            repoDir = repoDir.slice(1);
          }
        } else if (pdfDir.startsWith("data/")) {
          repoDir = pdfDir.replace(/^data\//, "");
        }

        await mkdir(repoDir, { recursive: true, baseDir: BaseDirectory.Data });

        for (const file of files) {
          const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
          if (!ALLOWED_EXTS.includes(ext)) {
            toast(`Unsupported file type: ${file.name}`, "info");
            continue;
          }

          // Calculate SHA256 hash and check for duplicates
          const sha256 = await calculateSHA256(file);
          const hashCheck = await checkPaperHash(sha256);
          
          if (hashCheck.exists) {
            skipped.push(file.name);
            continue;
          }

          // Handle Firefox PDF documents by extracting the actual PDF content
          let fileData: Uint8Array;
          const extractedPdf = await extractPdfFromFirefoxDocument(file);
          
          if (extractedPdf) {
            console.log(`Extracted PDF content from Firefox document: ${file.name}`);
            fileData = extractedPdf;
            // Ensure the filename ends with .pdf if it doesn't already
            const finalFilename = file.name.toLowerCase().endsWith('.pdf') ? file.name : `${file.name}.pdf`;
            await writeFile(`${repoDir}/${finalFilename}`, fileData, {
              baseDir: BaseDirectory.Data,
            });
          } else {
            // Regular file - use as-is
            const buf = await file.arrayBuffer();
            fileData = new Uint8Array(buf);
            await writeFile(`${repoDir}/${file.name}`, fileData, {
              baseDir: BaseDirectory.Data,
            });
          }
          copied.push(file.name);
        }
      } else {
        // Web mode: Upload files to the backend server
        for (const file of files) {
          const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
          if (!ALLOWED_EXTS.includes(ext)) {
            toast(`Unsupported file type: ${file.name}`, "info");
            continue;
          }

          // Upload file to backend (backend handles hash checking)
          const formData = new FormData();
          formData.append('file', file);
          
          const response = await fetch('/upload-paper', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Failed to upload ${file.name}: ${response.statusText}`);
          }

          const result = await response.json();
          if (result.duplicate) {
            skipped.push(file.name);
          } else {
            copied.push(file.name);
          }
        }
      }

      // Provide feedback to user
      if (copied.length) {
        toast(`📄 Added ${copied.length} file(s) to repository`, "success");
      }
      if (skipped.length) {
        toast(`⚠️ Skipped ${skipped.length} duplicate file(s): ${skipped.join(', ')}`, "info");
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
    onSuccess: async (id: string) => {
      setTaskId(id);
      localStorage.setItem('ingestion-task-id', id);
      // Save the pdfDir to backend config
      try {
        await fetch(`${API_BASE}/pdf_dir`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pdf_dir: pdfDir })
        });
      } catch (err) {
        console.error('Failed to save pdf_dir config:', err);
      }
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
        localStorage.removeItem('ingestion-task-id');
      }
    }
  }, [isError, taskErr]);

  // Clear taskId from localStorage when task completes or fails
  useEffect(() => {
    if (taskData && ((taskData as any).status === "done" || (taskData as any).status === "error")) {
      // Show toast notification for completion
      if ((taskData as any).status === "done") {
        const progressMsg = (taskData as any).progress?.message || "Ingestion completed successfully!";
        toast(`✅ ${progressMsg}`, "success");
      } else if ((taskData as any).status === "error") {
        toast(`❌ Ingestion failed: ${(taskData as any).error}`, "error");
      }
      
      // Clear the task ID after a short delay to let user see the final status
      setTimeout(() => {
        setTaskId(null);
        localStorage.removeItem('ingestion-task-id');
      }, 3000);
    }
  }, [taskData, toast]);

  // ───────────────────────────────────────── Resolve default repo dir ──
  useEffect(() => {
    // Initialise the default PDF directory to the app's data dir on Desktop.
    (async () => {
      try {
        const w = window as any;
        const isTauri =
          Boolean(w?.__TAURI__) ||
          Boolean(w?.__TAURI_INTERNALS__) ||
          Boolean(w?.isTauri) ||
          navigator.userAgent.includes("Tauri");

        if (isTauri) {
          const { dataDir, join } = await import("@tauri-apps/api/path");
          const base = await dataDir();
          const abs = await join(base, "papers");
          setPdfDir(abs);
        } else {
          // Web preview fallback – retain historical relative path
          setPdfDir("data/papers");
        }
      } catch {
        // On any failure, fall back to previous relative default
        setPdfDir("data/papers");
      }
    })();
  }, []);

  // ───────────────────────────────────────── Browse handler ──
  const handleBrowse = async () => {
    try {
      const result = await invoke<string | null>('browse_parent_folder', { 
        currentPath: pdfDir || null 
      });
      
      if (result) {
        setPdfDir(result);
        toast("Directory selected successfully!", "success");
      }
      // If result is null, user cancelled - no message needed
    } catch (error) {
      console.error("Folder picker error:", error);
      toast("Failed to open folder picker", "error");
    }
  };

  const handleShowFiles = async () => {
    try {
      if (!pdfDir.trim()) {
        toast("Please enter a directory path first", "info");
        return;
      }
      
      await invoke('open_folder_in_explorer', { path: pdfDir.trim() });
      toast("Opened folder in Explorer", "success");
    } catch (error) {
      console.error("Explorer open error:", error);
      toast(`Failed to open folder: ${String(error)}`, "error");
    }
  };

  // ───────────────────────────────────────── Path validation ──
  const [pathValidation, setPathValidation] = useState<{ isValid: boolean; message: string } | null>(null);

  // ───────────────────────────────────────── Clear database ──
  const [clearDbConfirm, setClearDbConfirm] = useState(false);

  const clearDbMutation = useMutation<void, Error, void>({
    mutationFn: clearDatabase,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["papers"] });
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      queryClient.invalidateQueries({ queryKey: ["insights"] });
      setClearDbConfirm(false);
      toast("🗑️ Database cleared successfully", "success");
    },
    onError: (error) => {
      toast(`Failed to clear database: ${String(error)}`, "error");
      setClearDbConfirm(false);
    },
  });
  
  const validatePath = async (path: string) => {
    if (!path.trim()) {
      setPathValidation(null);
      return;
    }
    
    try {
      const isValid = await invoke<boolean>('validate_directory', { path: path.trim() });
      setPathValidation({
        isValid,
        message: isValid ? "✓ Valid directory" : "✗ Directory not found"
      });
    } catch (error) {
      setPathValidation({
        isValid: false,
        message: "✗ Unable to validate path"
      });
    }
  };

  // Debounced validation
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      validatePath(pdfDir);
    }, 500);
    
    return () => clearTimeout(timeoutId);
  }, [pdfDir]);

  // On mount, try to load the last-used pdfDir from backend config
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/pdf_dir`);
        if (resp.ok) {
          const data = await resp.json();
          if (data.pdf_dir) setPdfDir(data.pdf_dir);
        }
      } catch (err) {
        // Ignore if not present
      }
    })();
  }, []);

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
        <div className="flex gap-2">
          <div className="relative flex-1">
            <input
              className="border rounded px-2 py-1 pr-8 w-full text-sm bg-headerBg text-primaryText"
              type="text"
              value={pdfDir}
              onChange={(e) => setPdfDir(e.target.value)}
              placeholder="Enter papers directory path..."
            />
            <button
              className="absolute right-1 top-1/2 transform -translate-y-1/2 p-1 hover:bg-gray-200 rounded"
              onClick={handleBrowse}
              title="Browse for directory (opens in parent folder for context)"
            >
              <MagnifyIcon className="w-4 h-4 text-gray-600" />
            </button>
          </div>
        </div>
        {pathValidation && (
          <div className={`text-xs ${pathValidation.isValid ? 'text-green-600' : 'text-red-600'}`}>
            {pathValidation.message}
          </div>
        )}
      </div>
      <div className="flex gap-2">
        <button
          className="flex-1 bg-accent hover:bg-accentHover text-white px-4 py-2 rounded font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={() => ingestMutation.mutate(pdfDir)}
          disabled={ingestMutation.isPending || !pdfDir.trim() || Boolean(taskId && (!taskData || (taskData as any).status === "running" || (taskData as any).status === "pending"))}
        >
          {ingestMutation.isPending ? "Processing..." : 
           (taskId && (!taskData || (taskData as any).status === "running" || (taskData as any).status === "pending")) ? "Ingestion in progress..." : 
           "Ingest PDFs"}
        </button>
                  <button
            className="px-3 py-1 rounded bg-buttonBg text-defaultText"
            onClick={handleShowFiles}
            title="Show files in directory (opens in Explorer)"
          >
            📁
          </button>
      </div>
      {taskId && (
        <div className="space-y-2">
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
          
          {/* Progress bar – gray backdrop with dynamic filled width */}
          {(!taskData || (taskData as any).status === "running" || (taskData as any).status === "pending") && (
            <div className="w-full bg-gray-300 rounded-full h-2 overflow-hidden relative">
              {/* Filled portion reflects percentage of files processed */}
              <div
                className="absolute top-0 left-0 h-full bg-accentText rounded-full transition-all duration-300 ease-out"
                style={{ width: `${(taskData as any)?.progress?.percentage || 0}%` }}
              />
            </div>
          )}
          
          {/* Progress message */}
          {taskData && (taskData as any).progress && (
            <div className="text-xs text-gray-600 mt-1">
              {(taskData as any).progress.message}
            </div>
          )}
        </div>
      )}

      {/* Debug: Clear Database */}
      <div className="mt-6 pt-4 border-t border-gray-300">
        <h3 className="text-sm font-medium text-gray-600 mb-2">Debug</h3>
        {clearDbConfirm ? (
          <div className="flex items-center gap-2">
            <button
              className="px-3 py-1 rounded bg-red-600/50 hover:bg-red-600 text-white text-xs disabled:opacity-50"
              disabled={clearDbMutation.isPending}
              onClick={() => clearDbMutation.mutate()}
            >
              {clearDbMutation.isPending ? "Clearing..." : "Confirm Clear DB"}
            </button>
            <button
              className="px-2 py-1 rounded bg-gray-500 hover:bg-gray-600 text-white text-xs"
              onClick={() => setClearDbConfirm(false)}
              disabled={clearDbMutation.isPending}
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            className="px-3 py-1 rounded bg-buttonBg text-defaultText disabled:opacity-50 text-xs"
            disabled={clearDbMutation.isPending}
            onClick={() => setClearDbConfirm(true)}
          >
            Clear Database
          </button>
        )}
        <p className="text-xs text-gray-500 mt-1">
          ⚠️ This will permanently delete all papers, collections, insights, and SHA256 hashes
        </p>
      </div>
    </div>
  );
}