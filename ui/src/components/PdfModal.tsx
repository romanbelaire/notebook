import { useEffect, useState } from "react";
import { readFile, BaseDirectory } from "@tauri-apps/plugin-fs";
// @ts-ignore – react-pdf may not ship bundled TS types; ignore during build
// eslint-disable-next-line import/no-extraneous-dependencies
import { Document, Page, pdfjs } from "react-pdf";
import CloseIcon from "../assets/close.svg?react";

// Configure pdf.js worker CDN (avoids extra bundling complexity)
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

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
        // If running in Tauri (plugin available), load from app data; else fetch over HTTP
        const w = window as any;
        const isTauri =
          Boolean(w.__TAURI__) ||
          Boolean(w.__TAURI_INTERNALS__) ||
          Boolean(w.isTauri) ||
          navigator.userAgent.includes("Tauri");
        if (isTauri) {
          // 1) Load raw bytes
          const bytes = await readFile(`papers/${filename}`, {
            baseDir: BaseDirectory.Data,
          });

          // 2) Ensure Uint8Array (older runtimes may give number[])
          const buffer = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes as number[]);

          // Debug: inspect signature
          console.log("First 5 ASCII chars of file:", new TextDecoder("ascii").decode(buffer.slice(0, 5)));
          if (!looksLikePDF(buffer)) {
            throw new Error("Loaded file is not a PDF – verify path/baseDir");
          }

          setPdfData(buffer); // keep for future context-extraction features

          // 3) Create a Blob URL for react-pdf
          const blobUrl = URL.createObjectURL(
            new Blob([buffer], { type: "application/pdf" })
          );
          setPdfUrl(blobUrl);
        } else {
          const resp = await fetch(`/papers/${filename}`);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          if (resp.headers.get("content-type") !== "application/pdf") {
            throw new Error(
              `Unexpected content-type ${resp.headers.get("content-type")}. Not a PDF.`
            );
          }
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          setPdfUrl(url);
        }
      } catch (err) {
        console.error(`Failed to load PDF: ${err}`);
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
              <Page pageNumber={page} width={Math.min(800, window.innerWidth * 0.75)} />
            </Document>
          ) : (
            <div className="p-4">Fetching PDF…</div>
          )}
        </div>
        {numPages && (
          <footer className="p-2 border-t border-gray-300 dark:border-gray-700 flex items-center gap-2 text-sm">
            <button
              className="px-2 py-0.5 border rounded disabled:opacity-40"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
            >
              ◄
            </button>
            <span>{page} / {numPages}</span>
            <button
              className="px-2 py-0.5 border rounded disabled:opacity-40"
              onClick={() => setPage((p) => Math.min(numPages, p + 1))}
              disabled={page >= numPages}
            >
              ►
            </button>
          </footer>
        )}
      </div>
    </div>
  );
}