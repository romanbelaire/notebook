import { useEffect, useState } from "react";
import { readFile, BaseDirectory } from "@tauri-apps/plugin-fs";
// @ts-ignore – react-pdf may not ship bundled TS types; ignore during build
// eslint-disable-next-line import/no-extraneous-dependencies
import { Document, Page, pdfjs } from "react-pdf";
// Import react-pdf CSS to fix TextLayer warning
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import CloseIcon from "../assets/close.svg?react";

// Configure pdf.js worker CDN (avoids extra bundling complexity)
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

// Base backend URL (same env var used in api.ts)
const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface Props {
  /** Filename as stored in data/papers (e.g., "paper.pdf"). */
  filename: string | null;
  /** Close handler – clicking backdrop or ✖️ invokes this. */
  onClose: () => void;
  /** Page to open initially (1-based). */
  initialPage?: number;
}

// Quick sanity-check: first four bytes of any valid PDF are "%PDF"
function looksLikePDF(buf: Uint8Array) {
  return (
    buf.length >= 4 &&
    buf[0] === 0x25 && // %
    buf[1] === 0x50 && // P
    buf[2] === 0x44 && // D
    buf[3] === 0x46 // F
  );
}

/**
 * PdfModal displays a full-document viewer for a given PDF stored under
 * the ingested papers directory. It loads the file via Tauri's fs plugin,
 * converts it to a Uint8Array and renders it with react-pdf directly to
 * avoid encoding issues.
 */
export default function PdfModal({ filename, onClose, initialPage = 1 }: Props) {
  const [pdfData, setPdfData] = useState<Uint8Array | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>();
  const [page, setPage] = useState<number>(initialPage);

  // Load PDF bytes when filename changes
  useEffect(() => {
    async function load() {
      if (!filename) return;

      try {
        // Always try to fetch from the backend API first, which can search multiple locations
        const pdfUrlBackend = `${API_BASE}/papers/${filename}`;
        console.log(`[PDF] Attempting backend fetch: ${pdfUrlBackend}`);
        const resp = await fetch(pdfUrlBackend, { mode: "cors" });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        }
        
        const contentType = resp.headers.get("content-type");
        if (contentType !== "application/pdf") {
          throw new Error(`Unexpected content-type ${contentType}. Expected application/pdf.`);
        }
        
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        setPdfUrl(url);
        
        // Also read as array buffer for potential future context extraction
        const arrayBuffer = await blob.arrayBuffer();
        setPdfData(new Uint8Array(arrayBuffer));
        
      } catch (err) {
        console.error(`Failed to load PDF: ${err}`);
        
        // Fallback: try Tauri file system if we're in Tauri mode
        const w = window as any;
        const isTauri =
          Boolean(w.__TAURI__) ||
          Boolean(w.__TAURI_INTERNALS__) ||
          Boolean(w.isTauri) ||
          navigator.userAgent.includes("Tauri");
          
        if (isTauri) {
          try {
            console.log("Trying Tauri fallback...");

            let bytes: Uint8Array | number[];

            try {
              // Prefer absolute path resolution via @tauri-apps/api/path to avoid
              // ambiguities with BaseDirectory lookup precedence.
              const { dataDir, join } = await import("@tauri-apps/api/path");
              const base = await dataDir();
              const papersDir = await join(base, "papers");
              const absPath = await join(papersDir, filename);
              try {
                const tauriFs = await import("@tauri-apps/plugin-fs");
                if ("readDir" in tauriFs) {
                  // @ts-ignore – dynamic import property access
                  const dirEntries = await tauriFs.readDir(papersDir);
                  console.log(`[PDF] Absolute papers dir: ${papersDir} – contains ${dirEntries.length} entries`);
                } else {
                  console.log(`[PDF] readDir not available; skipping directory listing for ${papersDir}`);
                }
              } catch (dirErr) {
                console.warn(`[PDF] Could not list directory ${papersDir}:`, dirErr);
              }
              console.log(`[PDF] Trying absolute path: ${absPath}`);
              bytes = await readFile(absPath); // absolute read – no baseDir
            } catch (absErr) {
              // Fallback #2: try relative to Data base dir (legacy behaviour)
              console.warn("[PDF] Absolute read failed, trying baseDir fallback", absErr);
              try {
                const tauriFs = await import("@tauri-apps/plugin-fs");
                if ("readDir" in tauriFs) {
                  // @ts-ignore
                  const baseDirEntries = await tauriFs.readDir("papers", { baseDir: BaseDirectory.Data });
                  console.log(`[PDF] BaseDirectory.Data papers dir contains ${baseDirEntries.length} entries`);
                } else {
                  console.log("[PDF] readDir not available in browser build; skipping directory listing");
                }
              } catch (dirErr2) {
                console.warn("[PDF] Could not list BaseDirectory.Data papers dir:", dirErr2);
              }
              bytes = await readFile(`papers/${filename}`, {
                baseDir: BaseDirectory.Data,
              });
            }

            const buffer = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes as number[]);

            console.log("First 5 ASCII chars of file:", new TextDecoder("ascii").decode(buffer.slice(0, 5)));
            if (!looksLikePDF(buffer)) {
              throw new Error("Loaded file is not a PDF – verify path/baseDir");
            }

            setPdfData(buffer);

            const blobUrl = URL.createObjectURL(
              new Blob([buffer], { type: "application/pdf" })
            );
            setPdfUrl(blobUrl);
          } catch (tauriErr) {
            console.error(`Tauri fallback also failed: ${tauriErr}`);
          }
        }
      }
    }
    load();

    return () => {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    };
  }, [filename]);

  if (!filename) return null; // Safety – shouldn't be rendered without a target

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-900 w-[80vw] h-[90vh] rounded shadow-lg flex flex-col"
        onClick={(e) => e.stopPropagation()} // stop backdrop close when interacting inside
      >
        <header className="flex items-center justify-between p-2 border-b border-gray-300 dark:border-gray-700 text-sm">
          <span className="truncate max-w-[75%]" title={filename}>{filename}</span>
          <button onClick={onClose} className="w-6 h-6 p-0 flex items-center justify-center" aria-label="Close">
            <CloseIcon className="w-5 h-5 flex-shrink-0" />
          </button>
        </header>
        <div className="flex-1 overflow-auto flex justify-center">
          {pdfUrl ? (
            <Document
              file={pdfUrl}
              onLoadSuccess={({ numPages }: { numPages: number }) => setNumPages(numPages)}
              loading={<div className="p-4">Loading PDF…</div>}
            >
              <div className="space-y-4">
                {numPages && Array.from({ length: numPages }, (_, i) => (
                  <Page 
                    key={i + 1} 
                    pageNumber={i + 1} 
                    width={Math.min(800, window.innerWidth * 0.75)}
                    className="shadow-lg"
                  />
                ))}
              </div>
            </Document>
          ) : (
            <div className="p-4">Fetching PDF…</div>
          )}
        </div>
      </div>
    </div>
  );
}