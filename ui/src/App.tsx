import { useState, useEffect } from "react";
import ChatWindow from "./components/ChatWindow";
import Sidebar from "./components/Sidebar";
import ScratchPad from "./components/ScratchPad";
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
  const SIDEBAR_WIDTH = '18rem';
  const CLOSED_WIDTH = '0.0625rem'; // 1px to allow drop shadow visibility when collapsed
  const TOGGLE_SIZE = '2.5rem'; // matches Tailwind w-10 h-10 (2.5rem)
  const [arrowHover, setArrowHover] = useState(false);
  const theme = useSettingsStore((s) => s.theme);
  const toast = useToast();

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

  const renderTab = () => {
    switch (activeTab) {
      case "Chat":
        return <ChatWindow />;
      case "Notepad":
        return <ScratchPad />;
      case "Library":
        return <LibraryView />;
      case "Data":
        return <IngestView />;
      case "Settings":
        return <SettingsView />;
      default:
        return null;
    }
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

  return (
    <DndContext sensors={sensors}>
    <div className="app-frame relative h-screen w-screen flex flex-col bg-primaryBg text-defaultText" onDragOver={handleRootDragOver} onDropCapture={handleRootDrop}>

      {/* top margin handled via pseudo-element */}

      <header className="p-3 mt-5 border-b flex items-center justify-between bg-headerBg text-light border-primaryBg shadow app-drag select-none">
        <h1 className="title-brand text-2xl font-mono">Notebook</h1>
        {/* Left side placeholder to allow drag area */}
        <div className="flex-1" />
        <nav className="flex items-center gap-4 no-drag">
          {/* Segmented control */}
          <div className="relative grid bg-secondaryBg/60 rounded-full overflow-hidden h-10 shadow-inner" style={{ gridTemplateColumns: `repeat(${tabs.length}, 1fr)` }}>
            {/* animated slider */}
            <span
              className="absolute inset-0 m-0.5 bg-buttonBg rounded-full transition-transform duration-300 shadow"
              style={{ width: `${100 / tabs.length}%`, transform: `translateX(${tabs.indexOf(activeTab) * 100}%)` }}
            />
            {tabs.map((t) => (
              <button
                key={t}
                className={
                  "relative z-10 w-full h-full flex items-center justify-center text-sm bg-transparent border-none focus:outline-none transition-colors" +
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
      </header>

      {/* Floating window-controls overlay – keeps them visible on narrow widths */}
      <div className="no-drag absolute" style={{ top: '4px', right: '4px', zIndex: 100 }}>
        <WindowControls />
      </div>

      <div className="flex flex-1 overflow-hidden relative">
        <div
          className={
            `relative h-full transition-all duration-300 ease-in-out overflow-hidden pointer-events-none border-r border-primaryBg bg-secondaryBg ${arrowHover ? 'shadow-[0_0_10px_rgba(255,255,255,0.2)]' : 'shadow-md'} transition-shadow`
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
          {renderTab()}
        </main>
      </div>
      <InsightModal />
    </div>
    </DndContext>
  );
}

export default App
