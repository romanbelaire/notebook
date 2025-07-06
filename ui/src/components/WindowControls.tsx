import { useEffect, useState } from "react";
import MinimizeIcon from "../assets/minimize.svg?react";
import MaximizeIcon from "../assets/maximize.svg?react";
import WindowedIcon from "../assets/windowed.svg?react";
import CloseIcon from "../assets/close.svg?react";

/**
 * WindowControls renders the typical Windows control buttons (minimize, maximize/restore, close)
 * for a borderless Tauri window. All Tauri API calls are dynamically imported at runtime so that
 * the component is tree-shaken from the browser build where the APIs are absent.
 */
export default function WindowControls() {
  const [isMax, setIsMax] = useState(false);

  // Track maximize / unmaximize events (Tauri desktop build only)
  useEffect(() => {
    // Lightweight runtime check – avoid importing @tauri-apps/api in the web build.
    const w = window as any;
    const isTauri =
      Boolean(w.__TAURI__) ||
      Boolean(w.__TAURI_INTERNALS__) ||
      Boolean(w.isTauri) ||
      navigator.userAgent.includes("Tauri");

    if (!isTauri) return;

    let unlistenMax: (() => void) | undefined;
    let unlistenUnmax: (() => void) | undefined;

    (async () => {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();

      setIsMax(await win.isMaximized());

      unlistenMax = await win.listen("tauri://maximize", () => {
        setIsMax(true);
      });
      unlistenUnmax = await win.listen("tauri://unmaximize", () => {
        setIsMax(false);
      });
    })();

    return () => {
      unlistenMax?.();
      unlistenUnmax?.();
    };
  }, []);

  async function handle(action: "minimize" | "maximizeToggle" | "close") {
    const w = window as any;
    const isTauri =
      Boolean(w.__TAURI__) ||
      Boolean(w.__TAURI_INTERNALS__) ||
      Boolean(w.isTauri) ||
      navigator.userAgent.includes("Tauri");
    if (!isTauri) return;

    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const appWindow = getCurrentWindow();

    switch (action) {
      case "minimize": {
        await appWindow.minimize();
        return;
      }
      case "maximizeToggle": {
        const currentlyMax = await appWindow.isMaximized();
        if (currentlyMax) {
          await appWindow.unmaximize();
          setIsMax(false);
        } else {
          await appWindow.maximize();
          setIsMax(true);
        }
        return;
      }
      case "close": {
        await appWindow.close();
        return;
      }
    }
  }

  const btnClass =
    "no-drag w-8 h-8 p-0 flex items-center justify-center text-light border-none bg-transparent hover:bg-buttonBg rounded transition-colors focus:outline-none focus:ring-0";

  return (
    <div className="flex items-center gap-1 no-drag select-none">
      <button
        aria-label="Minimize"
        className={btnClass}
        onClick={() => handle("minimize")}
      >
        <MinimizeIcon className="w-3 h-3" />
      </button>
      <button
        aria-label={isMax ? "Restore" : "Maximize"}
        className={btnClass}
        onClick={() => handle("maximizeToggle")}
      >
        {isMax ? (
          <WindowedIcon className="w-3 h-3" />
        ) : (
          <MaximizeIcon className="w-3 h-3" />
        )}
      </button>
      <button
        aria-label="Close"
        className={`${btnClass} hover:bg-red-600 hover:text-white`}
        onClick={() => handle("close")}
      >
        <CloseIcon className="w-3 h-3" />
      </button>
    </div>
  );
} 