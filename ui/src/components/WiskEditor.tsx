import { useEffect, useRef, forwardRef, useImperativeHandle, useCallback } from 'react';
import { getWiskAssetPath, WISK_CONFIG } from '../config/wisk';

// Types
export interface WiskElement {
  id: string;
  component: string;
  value: any;
}

export interface WiskDocument {
  data: {
    config: {
      name: string;
      theme: string;
      plugins: string[];
    };
    elements: WiskElement[];
    pluginData: Record<string, any>;
  };
}

export interface WiskEditorProps {
  document?: WiskDocument;
  onChange?: (document: WiskDocument) => void;
  readonly?: boolean;
  theme?: string;
  plugins?: string[];
  className?: string;
  style?: React.CSSProperties;
  backendUrl?: string;
  enableAI?: boolean;
  placeholder?: string;
}

export interface WiskEditorRef {
  getDocument: () => WiskDocument | null;
  setDocument: (doc: WiskDocument) => void;
  focus: () => void;
  addBlock: (type: string, value?: any, afterElementId?: string) => string;
  deleteBlock: (elementId: string) => void;
  updateBlock: (elementId: string, value: any) => void;
  convertFromHtml: (html: string) => WiskDocument;
  convertToHtml: () => string;
}

// Global wisk object interface
declare global {
  interface Window {
    wisk: {
      editor: {
        document: WiskDocument;
        readonly: boolean;
        backendUrl: string;
        createNewBlock: (elementId: string, blockType: string, value: any, focusIdentifier?: any) => string;
        deleteBlock: (elementId: string) => void;
        updateBlock: (elementId: string, key: string, value: any, rec?: string) => void;
        focusBlock: (elementId: string, identifier?: any) => void;
        generateNewId: (prefix?: string) => string;
        addConfigChange: (id: string, value: any) => Promise<void>;
      };
      plugins: {
        loadPlugin: (pluginName: string) => Promise<void>;
        loadedPlugins: string[];
        defaultPlugins: string[];
        pluginData: Record<string, any>;
      };
      theme: {
        setTheme: (theme: string) => void;
      };
      sync: {
        saveUpdates: () => Promise<void>;
      };
    };
  }
}

// Global flag to track if Wisk assets are loaded
let wiskAssetsLoaded = false;
const loadedScripts = new Set<string>();

// Utility functions
async function loadWiskAssets(baseUrl = getWiskAssetPath()): Promise<void> {
  // If assets are already loaded and wisk object exists, skip loading
  if (wiskAssetsLoaded && (window as any).wisk && (window as any).BaseTextElement) {
    return;
  }

  const assetsToLoad = [
    // CSS files
    { type: 'css', url: `${baseUrl}/css/mini-dialog.css` },
    { type: 'css', url: `${baseUrl}/css/right-sidebar.css` },
    { type: 'css', url: `${baseUrl}/css/left-sidebar.css` },
    { type: 'css', url: `${baseUrl}/style.css` },
    { type: 'css', url: `${baseUrl}/global.css` },
    { type: 'css', url: `${baseUrl}/js/theme/variables.css` },
    
    // Core JS files - IMPORTANT: Load in correct order
    // wisk.js must load before utils.js because utils.js depends on wisk
    // base-text-element.js must load before plugins.js because other plugins extend it
    // lit-core must load before plugins that use LitElement
    // selector-element.js must load before editor.js because editor.js uses it
    // theme.js must load before plugins that use wisk.theme.getThemes() (like options-component)
    { type: 'js', url: `${baseUrl}/global.js` },
    { type: 'js', url: `${baseUrl}/js/polyfill.js` },
    { type: 'js', url: `${baseUrl}/js/wisk.js` },
    { type: 'js', url: `${baseUrl}/js/utils.js` },
    // Load theme.js BEFORE plugins so wisk.theme.getThemes() is available
    { type: 'js', url: `${baseUrl}/js/theme/theme.js` },
    // Load selector-element BEFORE editor.js because editor.js calls showSelector
    { type: 'js', url: `${baseUrl}/js/elements/selector-element.js` },
    { type: 'js', url: `${baseUrl}/js/editor.js` },
    // Load lit-core BEFORE plugins that need it (like neo-ai, search-element, etc.)
    { type: 'js', url: `${baseUrl}/a7/cdn/lit-core-2.7.4.min.js`, loadAsModule: true },
    // Load base-text-element.js BEFORE plugins.js so BaseTextElement class is available
    // This prevents "BaseTextElement is not defined" errors
    { type: 'js', url: `${baseUrl}/js/plugins/code/base-text-element.js` },
    { type: 'js', url: `${baseUrl}/js/plugins/plugins.js` },
  ];

  // Load assets sequentially to ensure proper order (especially for JS)
  for (const asset of assetsToLoad) {
    await new Promise<void>((resolve, reject) => {
      if (asset.type === 'css') {
        if (document.querySelector(`link[href="${asset.url}"]`)) {
          resolve();
          return;
        }
        
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = asset.url;
        link.onload = () => resolve();
        link.onerror = () => reject(new Error(`Failed to load CSS: ${asset.url}`));
        document.head.appendChild(link);
      } else if (asset.type === 'js') {
        // Check if script is already loaded - prevent redeclaration errors
        if (loadedScripts.has(asset.url)) {
          resolve();
          return;
        }
        
        const existingScript = document.querySelector(`script[src="${asset.url}"]`);
        if (existingScript) {
          loadedScripts.add(asset.url);
          resolve();
          return;
        }
        
        const script = document.createElement('script');
        script.src = asset.url;
        // If loadAsModule is true, set type to module for ES6 imports
        if ((asset as any).loadAsModule) {
          script.type = 'module';
        }
        script.onload = () => {
          loadedScripts.add(asset.url);
          // If this is base-text-element.js, mark it as loaded in wisk.plugins to prevent reloading
          if (asset.url.includes('base-text-element.js') && window.wisk?.plugins) {
            // Mark base-text-element as loaded if it's tracked as a plugin
            if (!window.wisk.plugins.loadedPlugins.includes('base-text-element')) {
              window.wisk.plugins.loadedPlugins.push('base-text-element');
            }
          }
          // Add a small delay to ensure script execution completes
          setTimeout(() => resolve(), 50);
        };
        script.onerror = () => reject(new Error(`Failed to load JS: ${asset.url}`));
        document.head.appendChild(script);
      }
    });
  }

  wiskAssetsLoaded = true;
}

function htmlToWiskDocument(html: string): WiskDocument {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const elements: WiskElement[] = [];

  const textNodes = doc.querySelectorAll('p, h1, h2, h3, h4, h5, h6, div');
  
  textNodes.forEach((node, index) => {
    let component = 'text-element';
    
    if (node.tagName === 'H1') component = 'heading1-element';
    else if (node.tagName === 'H2') component = 'heading2-element';
    else if (node.tagName === 'H3') component = 'heading3-element';
    else if (node.tagName === 'H4') component = 'heading4-element';
    else if (node.tagName === 'H5') component = 'heading5-element';
    else if (node.tagName === 'H6') component = 'heading5-element';
    
    if (node.textContent?.trim()) {
      elements.push({
        id: `element_${index}_${Date.now()}`,
        component,
        value: {
          textContent: node.textContent || '',
          html: node.innerHTML || '',
        }
      });
    }
  });

  return {
    data: {
      config: {
        name: 'Converted Document',
        theme: 'default',
        plugins: ['text-element', 'heading1-element', 'heading2-element', 'heading3-element'],
      },
      elements,
      pluginData: {},
    }
  };
}

function wiskDocumentToHtml(doc: WiskDocument): string {
  if (!doc?.data?.elements) return '';
  
  let html = '';
  
  doc.data.elements.forEach((element) => {
    const value = element.value || {};
    
    switch (element.component) {
      case 'heading1-element':
        html += `<h1>${value.textContent || ''}</h1>`;
        break;
      case 'heading2-element':
        html += `<h2>${value.textContent || ''}</h2>`;
        break;
      case 'heading3-element':
        html += `<h3>${value.textContent || ''}</h3>`;
        break;
      case 'heading4-element':
        html += `<h4>${value.textContent || ''}</h4>`;
        break;
      case 'heading5-element':
        html += `<h5>${value.textContent || ''}</h5>`;
        break;
      case 'text-element':
      default:
        html += `<p>${value.textContent || ''}</p>`;
        break;
    }
  });
  
  return html;
}

const WiskEditor = forwardRef<WiskEditorRef, WiskEditorProps>(({
  document: initialDocument,
  onChange,
  readonly = false,
  theme = 'default',
  plugins = [],
  className = '',
  style = {},
  backendUrl = '',
  enableAI = false,
  placeholder = 'Write something or press \'/\' for commands'
}, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const initializedRef = useRef(false);
  const documentRef = useRef<WiskDocument | null>(null);

  // Initialize Wisk when component mounts
  useEffect(() => {
    // Use a global flag to prevent multiple initializations
    // Check multiple conditions to ensure Wisk is truly ready
    // Check if plugins are already loaded to prevent redeclaration errors
    const isWiskReady = (window as any).__wiskInitialized && 
                        window.wisk && 
                        wiskAssetsLoaded &&
                        window.wisk.plugins?.loadedPlugins?.length > 0;
    
    if (isWiskReady) {
      initializedRef.current = true;
      // If Wisk is already initialized, configure it for this instance
      window.wisk.editor.readonly = readonly;
      window.wisk.editor.backendUrl = backendUrl;
      if (window.wisk.theme?.setTheme) {
        window.wisk.theme.setTheme(theme);
      }
      
      // Wait for containers to exist, then initialize editor if needed
      const checkAndInit = () => {
        if (!containerRef.current || !editorRef.current) {
          return false;
        }
        
        const editorEl = containerRef.current.querySelector('#editor') as HTMLDivElement;
        if (!editorEl) {
          return false;
        }
        
        // If editor already has content, ensure it's editable
        if (editorEl.hasChildNodes()) {
          window.wisk.editor.readonly = readonly;
          // Set up change listener if not already set up
          setupChangeListener();
          return true;
        }
        
        // Editor is empty, initialize it
        if ((window as any).initEditor) {
          // Set document if provided, otherwise use existing or create empty
          if (initialDocument) {
            window.wisk.editor.document = initialDocument;
            documentRef.current = initialDocument;
          } else if (!window.wisk.editor.document) {
            // Create empty document if none exists
            const emptyDoc: WiskDocument = {
              data: {
                config: {
                  name: 'Untitled Document',
                  theme,
                  plugins: window.wisk.plugins?.defaultPlugins || [],
                },
                elements: [{
                  id: window.wisk.editor.generateNewId(),
                  component: 'text-element',
                  value: { textContent: '' }
                }],
                pluginData: {},
              }
            };
            window.wisk.editor.document = emptyDoc;
            documentRef.current = emptyDoc;
          }
          
          (window as any).initEditor(window.wisk.editor.document).catch((err: any) => {
            console.warn('Failed to initialize editor:', err);
          });
          setupChangeListener();
          return true;
        }
        
        return false;
      };
      
      // Try immediately
      if (!checkAndInit()) {
        // Wait for DOM to be ready
        const checkDOM = setInterval(() => {
          if (checkAndInit()) {
            clearInterval(checkDOM);
          }
        }, 100);
        
        // Cleanup after reasonable timeout
        setTimeout(() => clearInterval(checkDOM), 5000);
        return () => clearInterval(checkDOM);
      }
      
      return;
    }
    
    // Prevent concurrent initializations
    if ((window as any).__wiskInitializing) {
      // Wait for the other initialization to complete
      const checkInitialized = setInterval(() => {
        if ((window as any).__wiskInitialized && window.wisk) {
          clearInterval(checkInitialized);
          initializedRef.current = true;
        }
      }, 100);
      return () => clearInterval(checkInitialized);
    }
    
    // Mark that we're initializing
    (window as any).__wiskInitializing = true;

    const initWisk = async () => {
      try {
        await loadWiskAssets();
        
        // Wait for wisk object to be available
        let attempts = 0;
        const maxAttempts = 50;
        
        while (!window.wisk && attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 100));
          attempts++;
        }
        
        if (!window.wisk) {
          throw new Error('Wisk failed to initialize');
        }
        
        // Wait for BaseTextElement and MainElement to be defined
        // These are now loaded explicitly before plugins.js, so they should be available
        // Check customElements registry since they're registered as custom elements
        attempts = 0;
        while (customElements.get('base-text-element') === undefined && attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 100));
          attempts++;
        }
        
        attempts = 0;
        while (customElements.get('main-element') === undefined && attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 100));
          attempts++;
        }
        
        // Wait for plugins to actually load and register custom elements
        // The warnings above are premature - plugins load asynchronously
        // So we'll wait a bit more for plugins to finish registering
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Check again after waiting - but don't warn if they're still not found
        // as plugins.js will load them asynchronously
        if (customElements.get('base-text-element') === undefined) {
          console.debug('base-text-element custom element not yet registered (may load via plugins.js)');
        }
        
        if (customElements.get('main-element') === undefined) {
          console.debug('main-element custom element not yet registered (may load via plugins.js)');
        }

        // Configure wisk
        window.wisk.editor.readonly = readonly;
        window.wisk.editor.backendUrl = backendUrl;
        
        // Initialize theme before setting it
        // initTheme() loads theme-data.json and populates wisk.theme.themeObject.themes
        // This must complete before plugins try to use wisk.theme.getThemes()
        if ((window as any).initTheme && typeof (window as any).initTheme === 'function') {
          await (window as any).initTheme();
        } else {
          // If initTheme wasn't called automatically, wait for themeObject to be populated
          let attempts = 0;
          const maxAttempts = 20;
          while ((!window.wisk.theme?.themeObject?.themes || window.wisk.theme.themeObject.themes.length === 0) && attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
          }
        }
        
        // Set theme after initialization
        if (window.wisk.theme?.setTheme) {
          window.wisk.theme.setTheme(theme);
        }

        // Initialize with document or create empty one
        if (initialDocument) {
          window.wisk.editor.document = initialDocument;
          documentRef.current = initialDocument;
        } else {
          const emptyDoc: WiskDocument = {
            data: {
              config: {
                name: 'Untitled Document',
                theme,
                plugins: window.wisk.plugins?.defaultPlugins || [],
              },
              elements: [{
                id: window.wisk.editor.generateNewId(),
                component: 'text-element',
                value: { textContent: '' }
              }],
              pluginData: {},
            }
          };
          window.wisk.editor.document = emptyDoc;
          documentRef.current = emptyDoc;
        }

        // Ensure plugin metadata is available before we attempt to load any plugins
        if (!window.wisk.plugins?.pluginData) {
          try {
            const res = await fetch(`${getWiskAssetPath()}/js/plugins/plugin-data.json`);
            if (res.ok) {
              window.wisk.plugins.pluginData = await res.json();
            } else {
              console.warn('Failed to fetch plugin-data.json:', res.statusText);
            }
          } catch (err) {
            console.warn('Error fetching plugin-data.json:', err);
          }
        }

        // Load default plugins - but only if they're not already loaded
        if (window.wisk.plugins?.defaultPlugins && window.wisk.plugins.pluginData) {
          for (const plugin of window.wisk.plugins.defaultPlugins) {
            // Skip if plugin is already loaded to prevent redeclaration errors
            if (window.wisk.plugins.loadedPlugins?.includes(plugin)) {
              continue;
            }
            try {
              await window.wisk.plugins.loadPlugin(plugin);
            } catch (error) {
              console.warn(`Failed to load plugin ${plugin}:`, error);
            }
          }
        }

        // Set up change monitoring
        setupChangeListener();

        initializedRef.current = true;
        (window as any).__wiskInitialized = true;
        (window as any).__wiskInitializing = false;

        // Initialize the editor with the document - this creates the DOM elements
        // Only call initEditor if the editor container exists and hasn't been initialized yet
        // Check if editor was already initialized globally
        const editorAlreadyInitialized = (window as any).__wiskEditorInitialized || 
                                         (containerRef.current?.querySelector('#editor')?.hasChildNodes() ?? false);
        
        if ((window as any).initEditor && containerRef.current && editorRef.current && !editorAlreadyInitialized) {
          const editorEl = containerRef.current.querySelector('#editor') as HTMLDivElement;
          // Check if editor is already initialized by looking for existing elements
          if (editorEl && !editorEl.hasChildNodes()) {
            // Wait a bit to ensure DOM is ready and containers exist
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Ensure selector-element is defined before initializing editor
            // selector-element is needed by editor.js for showSelector
            let attempts = 0;
            const maxAttempts = 20;
            while (customElements.get('selector-element') === undefined && attempts < maxAttempts) {
              await new Promise(resolve => setTimeout(resolve, 100));
              attempts++;
            }
            
            if (customElements.get('selector-element') === undefined) {
              console.warn('selector-element not found, editor may fail');
            }
            
            // Ensure required containers exist before initializing
            const miniDialogBody = document.querySelector('.mini-dialog-body');
            const rightSidebarBody = document.querySelector('.right-sidebar-body');
            const leftSidebarBody = document.querySelector('.left-sidebar-body');
            const selectorElement = document.querySelector('selector-element');
            
            if (miniDialogBody && rightSidebarBody && leftSidebarBody && selectorElement) {
              await (window as any).initEditor(window.wisk.editor.document);
              (window as any).__wiskEditorInitialized = true;
            } else {
              console.warn('Wisk containers not ready, skipping initEditor', {
                miniDialogBody: !!miniDialogBody,
                rightSidebarBody: !!rightSidebarBody,
                leftSidebarBody: !!leftSidebarBody,
                selectorElement: !!selectorElement
              });
            }
          }
        } else if (editorAlreadyInitialized) {
          // Editor already initialized, just update the document if needed
          if (window.wisk?.editor?.setDocument && initialDocument) {
            window.wisk.editor.setDocument(initialDocument);
          }
        }
      } catch (error) {
        console.error('Failed to initialize Wisk:', error);
        throw error;
      }
    };

    initWisk();
  }, [readonly, theme, plugins, backendUrl, enableAI, placeholder, initialDocument]);

  const setupChangeListener = useCallback(() => {
    if (!window.wisk?.editor || !onChange) return;

    // Simple polling for changes
    const checkForChanges = () => {
      if (window.wisk?.editor?.document && documentRef.current) {
        const currentDoc = window.wisk.editor.document;
        if (JSON.stringify(currentDoc) !== JSON.stringify(documentRef.current)) {
          documentRef.current = { ...currentDoc };
          onChange(currentDoc);
        }
      }
    };

    const interval = setInterval(checkForChanges, 500);
    return () => clearInterval(interval);
  }, [onChange]);

  // Expose methods via ref
  useImperativeHandle(ref, () => ({
    getDocument: () => {
      return window.wisk?.editor?.document || documentRef.current;
    },
    
    setDocument: (doc: WiskDocument) => {
      if (window.wisk?.editor) {
        window.wisk.editor.document = doc;
        documentRef.current = doc;
        if (onChange) {
          onChange(doc);
        }
      }
    },
    
    focus: () => {
      if (window.wisk?.editor?.document?.data?.elements?.[0]) {
        const firstElementId = window.wisk.editor.document.data.elements[0].id;
        window.wisk.editor.focusBlock(firstElementId);
      }
    },
    
    addBlock: (type: string, value: any = {}, afterElementId?: string) => {
      if (!window.wisk?.editor) return '';
      
      const targetElementId = afterElementId || 
        (window.wisk.editor.document?.data?.elements?.length > 0 
          ? window.wisk.editor.document.data.elements[window.wisk.editor.document.data.elements.length - 1].id 
          : '');
      
      return window.wisk.editor.createNewBlock(targetElementId, type, value);
    },
    
    deleteBlock: (elementId: string) => {
      if (window.wisk?.editor?.deleteBlock) {
        window.wisk.editor.deleteBlock(elementId);
      }
    },
    
    updateBlock: (elementId: string, value: any) => {
      if (window.wisk?.editor?.updateBlock) {
        window.wisk.editor.updateBlock(elementId, '', value);
      }
    },

    convertFromHtml: htmlToWiskDocument,
    
    convertToHtml: () => {
      const doc = window.wisk?.editor?.document || documentRef.current;
      return doc ? wiskDocumentToHtml(doc) : '';
    },
  }), [onChange]);

  // Ensure toast container exists for Wisk notifications
  useEffect(() => {
    if (!document.querySelector('.toast-container')) {
      const container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
  }, []);

  return (
    <div 
      ref={containerRef}
      className={`wisk-editor-container ${className}`}
      style={style}
    >
      {/* Required Wisk containers for plugins */}
      <div className="mini-dialog hidden">
        <div className="mini-dialog-bg"></div>
        <div className="mini-dialog-content">
          <div className="mini-dialog-sheet-holder-area"></div>
          <div className="mini-dialog-sheet-holder"></div>
          <div className="mini-dialog-header">
            <p className="mini-dialog-title">Title</p>
            <button className="mini-dialog-close">
              <img src={`${getWiskAssetPath()}/a7/iconoir/xmark.svg`} alt="Close" className="plugin-icon" draggable="false" />
            </button>
          </div>
          <div className="mini-dialog-body"></div>
        </div>
      </div>

      <div className="right-sidebar right-sidebar-hidden">
        <div className="right-sidebar-header"></div>
        <div className="right-sidebar-body"></div>
      </div>

      <div className="left-sidebar left-sidebar-hidden">
        <div className="left-sidebar-header">
          <button className="left-sidebar-close">
            <p className="left-sidebar-title">Title</p>
            <img src={`${getWiskAssetPath()}/a7/forget/max-sidebar.svg`} alt="Close" className="plugin-icon" />
          </button>
        </div>
        <div className="left-sidebar-body"></div>
      </div>

      <div className="main">
        <div className="mix">
          <nav id="nav" style={{ display: 'none' }}>
            <div className="nav-top-icons"></div>
            <div className="nav-plugins"></div>
          </nav>

          <div className="editor">
            <div className="editor-main" id="editor" ref={editorRef}></div>
            <div id="last-space"></div>
          </div>
        </div>
      </div>

      {/* Required Wisk components */}
      <toolbar-element id="formatting-toolbar"></toolbar-element>
      <selector-element></selector-element>
      <command-palette style={{ zIndex: 9999, position: 'fixed' }}></command-palette>
      <search-element style={{ zIndex: 9999, position: 'fixed' }}></search-element>
    </div>
  );
});

WiskEditor.displayName = 'WiskEditor';

export default WiskEditor; 