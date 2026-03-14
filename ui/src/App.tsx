import { useState, useEffect, useRef } from "react";
import ChatWindow from "./components/ChatWindow";
import Sidebar from "./components/Sidebar";
import WiskScratchPad from "./components/WiskScratchPad";
import LibraryView from "./components/LibraryView";
import IngestView from "./components/IngestView";
import SettingsView from "./components/SettingsView";
import InsightModal from "./components/InsightModal";
import ChevronLeftIcon from "./assets/chevron-left.svg?react";
import ChevronRightIcon from "./assets/chevron-right.svg?react";
import GearIcon from "./assets/gear.svg?react";
import WindowControls from "./components/WindowControls";
import { mkdir, writeFile, BaseDirectory } from "@tauri-apps/plugin-fs";
import type { DragEvent as ReactDragEvent } from "react";
import { DndContext, useSensor, useSensors, PointerSensor } from "@dnd-kit/core";
import "./themes.css";
import { useSettingsStore } from "./store/settings";
import { useToast } from "./components/ToastProvider";
import { useUIStore } from "./store/ui";

function App() {
  const tabs = ["Chat", "Notepad", "Library", "Data", "Settings"] as const;
  const activeTab = useUIStore((s) => s.activeTab);
  const setActiveTab = useUIStore((s) => s.setActiveTab);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isNarrow, setIsNarrow] = useState(window.innerWidth <= 800);
  const SIDEBAR_WIDTH = '18rem';
  const CLOSED_WIDTH = '0.0625rem'; // 1px to allow drop shadow visibility when collapsed
  const TOGGLE_SIZE = '2.5rem'; // matches Tailwind w-10 h-10 (2.5rem)
  const [arrowHover, setArrowHover] = useState(false);
  const theme = useSettingsStore((s) => s.theme);
  const toast = useToast();

  // Detect when window is narrow for nav wrapping
  useEffect(() => {
    const handleResize = () => {
      setIsNarrow(window.innerWidth <= 800);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    // Debug: list all loaded style sheets on initial render
    const sheets = Array.from(document.styleSheets).map((ss) => ss.href ?? '[inline]');
    console.log('🔎 Loaded style sheets:', sheets);
    console.log('🔎 Style sheet details:');
    Array.from(document.styleSheets).forEach((ss, i) => {
      const id = (ss.ownerNode as HTMLElement | null)?.getAttribute?.('data-vite-dev-id') ?? '(inline)';
      let sample = '';
      try {
        const rules = ss.cssRules ?? [];
        sample = Array.from(rules).slice(0, 3).map(r => '  ' + (r as CSSStyleRule).cssText).join('\n');
      } catch (err) {
        sample = '[cross-origin]';
      }
      console.log(`#${i} ${id}`, ss.href || '[inline]', '\n', sample);
    });
  }, []);

  // Apply theme class to <body>
  useEffect(() => {
    const body = document.body;
    // Remove any existing theme- class
    body.className = body.className
      .split(" ")
      .filter((c) => !c.startsWith("theme-"))
      .join(" ");
    if (theme && theme !== "standard") {
      body.classList.add(`theme-${theme}`);
    }
  }, [theme]);

  // Render all tabs but hide inactive ones - this prevents unmounting/remounting
  // which causes Wisk to re-initialize and lose state
  const renderAllTabs = () => {
    return (
      <>
        <div style={{ display: activeTab === "Chat" ? "block" : "none" }}>
          <ChatWindow />
        </div>
        <div style={{ display: activeTab === "Notepad" ? "block" : "none" }}>
          <WiskScratchPad />
        </div>
        <div style={{ display: activeTab === "Library" ? "block" : "none" }}>
          <LibraryView />
        </div>
        <div style={{ display: activeTab === "Data" ? "block" : "none" }}>
          <IngestView />
        </div>
        <div style={{ display: activeTab === "Settings" ? "block" : "none" }}>
          <SettingsView />
        </div>
      </>
    );
  };

  const handleRootDragOver = (e: ReactDragEvent<HTMLDivElement>) => {
    // Allow dropping files (but don't interfere with other drag types)
    if (Array.from(e.dataTransfer.types).includes("Files")) {
      try {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
      } catch {
        /* readonly dropEffect – ignore */
      }
    }
  };

  const handleRootDrop = async (e: ReactDragEvent<HTMLDivElement>) => {
    // Only handle PDF files – defer everything else to child handlers
    const files = Array.from(e.dataTransfer.files ?? []);
    const pdf = files.find(
      (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")
    );
    if (!pdf) return; // not a PDF → let other handlers run

    // It's a PDF – consume event so descendants (e.g., ScratchPad) don't process it
    e.preventDefault();
    e.stopPropagation();

    try {
      // Detect if we're running in Tauri or web mode
      const w = window as any;
      const isTauri =
        Boolean(w?.__TAURI__) ||
        Boolean(w?.__TAURI_INTERNALS__) ||
        Boolean(w?.isTauri) ||
        navigator.userAgent.includes("Tauri");

      if (isTauri) {
        // Tauri mode: Use Tauri file system API
        await mkdir("papers", { recursive: true, baseDir: BaseDirectory.Data });

        const buf = await pdf.arrayBuffer();
        await writeFile(`papers/${pdf.name}`, new Uint8Array(buf), {
          baseDir: BaseDirectory.Data,
        });
      } else {
        // Web mode: Upload to backend server
        const formData = new FormData();
        formData.append('file', pdf);
        
        const response = await fetch('/upload-paper', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Failed to upload ${pdf.name}: ${response.statusText}`);
        }
      }

      // Switch to the Data tab for visual confirmation
      setActiveTab("Data");

      // Optionally notify success
      toast(`📄 Added '${pdf.name}' to paper repository!`, "success");
    } catch (err) {
      toast(`Failed to add PDF: ${String(err)}`, "error");
      throw err;
    }
  };

  // ───────────────────────────────────────── DnD-kit sensors ──
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }));

  // Nav component to reuse
  const navComponent = (
    <nav className="app-nav-tabs no-drag header-nav">
      {/* Segmented control */}
      <div className="relative grid bg-secondaryBg/60 rounded-full overflow-hidden h-10 shadow-inner app-nav-control" style={{ gridTemplateColumns: `repeat(${tabs.length}, 1fr)`, width: 'fit-content', display: 'grid' }}>
        {/* animated slider */}
        <span
          className="absolute inset-0 m-0.5 bg-buttonBg rounded-full transition-transform duration-300 shadow"
          style={{ width: `${100 / tabs.length}%`, transform: `translateX(${tabs.indexOf(activeTab) * 100}%)` }}
        />
        {tabs.map((t) => (
          <button
            key={t}
            className={
              "relative z-10 w-full h-full flex items-center justify-center text-sm bg-transparent border-none focus:outline-none transition-colors px-4 whitespace-nowrap" +
              (t === activeTab ? " text-accentText" : " text-light/80 hover:text-light")
            }
            onClick={() => setActiveTab(t)}
          >
            {t === "Settings" ? (
              <GearIcon className="w-4 h-4" />
            ) : (
              t
            )}
          </button>
        ))}
      </div>
    </nav>
  );

  return (
    <DndContext sensors={sensors}>
    <div className="app-frame relative h-screen w-screen flex flex-col bg-primaryBg text-defaultText" onDragOver={handleRootDragOver} onDropCapture={handleRootDrop}>

      {/* top margin handled via pseudo-element */}

      <header className="p-3 border-b bg-headerBg text-light border-primaryBg shadow app-drag select-none">
        {/* First row: Title and Controls (always together) */}
        <div className="flex items-center justify-between gap-2 w-full">
          <h1 className="title-brand text-2xl font-mono flex-shrink-0">Notebook</h1>
          {!isNarrow && navComponent}
          <div className="no-drag flex-shrink-0">
            <WindowControls />
          </div>
        </div>
        
        {/* Second row: Nav (only when narrow) */}
        {isNarrow && (
          <div className="flex justify-center w-full mt-2">
            {navComponent}
          </div>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden relative">
        <div
          className={
            `relative h-full transition-all duration-300 ease-in-out overflow-hidden border-r border-primaryBg bg-secondaryBg ${arrowHover ? 'shadow-[0_0_10px_rgba(255,255,255,0.2)]' : 'shadow-md'} transition-shadow ${sidebarOpen ? 'pointer-events-auto' : 'pointer-events-none'}`
          }
          style={{ width: sidebarOpen ? SIDEBAR_WIDTH : CLOSED_WIDTH }}
        >
          <Sidebar open={sidebarOpen} />
        </div>

        {/* Toggle button now external to sidebar for reliable hit area */}
        <button
          aria-label="Toggle sidebar"
          className={`no-drag absolute top-2 w-10 h-10 p-0 transition-all duration-300 ease-in-out bg-transparent border-none flex items-center justify-center rounded-full transition-opacity z-40 text-defaultText focus:outline-none focus:ring-0 cursor-pointer ${arrowHover ? 'drop-shadow-[0_0_6px_rgba(255,255,255,0.8)]' : ''} ${sidebarOpen ? 'hover:opacity-100' : 'opacity-30 hover:opacity-100'}`}
          style={{ left: sidebarOpen ? `calc(${SIDEBAR_WIDTH} - ${TOGGLE_SIZE})` : '0.5rem' }}
          onMouseEnter={() => setArrowHover(true)}
          onMouseLeave={() => setArrowHover(false)}
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          {sidebarOpen ? (
            <ChevronLeftIcon className="w-8 h-8 pointer-events-none text-accentText" />
          ) : (
            <ChevronRightIcon className="w-8 h-8 pointer-events-none text-accentText" />
          )}
        </button>

        <main className="flex-1 overflow-auto">
          {renderAllTabs()}
        </main>
      </div>
      <InsightModal />
    </div>
    </DndContext>
  );
}

export default App
