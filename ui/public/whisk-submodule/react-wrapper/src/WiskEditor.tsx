import React, { useEffect, useRef, forwardRef, useImperativeHandle, useCallback } from 'react';
import { WiskEditorProps, WiskEditorRef, WiskDocument } from './types';
import { loadWiskAssets, initializeWisk } from './utils';

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
    if (initializedRef.current) return;

    const initWisk = async () => {
      try {
        // Load Wisk assets (CSS, JS)
        await loadWiskAssets();
        
        // Initialize the global wisk object
        await initializeWisk({
          readonly,
          backendUrl,
          enableAI,
        });

        // Set theme
        if (window.wisk?.theme?.setTheme) {
          window.wisk.theme.setTheme(theme);
        }

        // Load additional plugins
        for (const plugin of plugins) {
          if (window.wisk?.plugins?.loadPlugin) {
            await window.wisk.plugins.loadPlugin(plugin);
          }
        }

        // Set initial document
        if (initialDocument) {
          setDocumentInternal(initialDocument);
        } else {
          // Create empty document
          const emptyDoc: WiskDocument = {
            data: {
              config: {
                name: 'Untitled Document',
                theme,
                plugins: [...window.wisk.plugins.defaultPlugins, ...plugins],
              },
              elements: [{
                id: window.wisk.editor.generateNewId(),
                component: 'text-element',
                value: { textContent: '' }
              }],
              pluginData: {},
            }
          };
          setDocumentInternal(emptyDoc);
        }

        // Set up change listener
        setupChangeListener();

        initializedRef.current = true;
      } catch (error) {
        console.error('Failed to initialize Wisk:', error);
        throw error;
      }
    };

    initWisk();
  }, [readonly, theme, plugins, backendUrl, enableAI, placeholder]);

  // Set up document change listener
  const setupChangeListener = useCallback(() => {
    if (!window.wisk?.editor) return;

    // Override the editor's update methods to capture changes
    const originalUpdateBlock = window.wisk.editor.updateBlock;
    const originalCreateNewBlock = window.wisk.editor.createNewBlock;
    const originalDeleteBlock = window.wisk.editor.deleteBlock;

    window.wisk.editor.updateBlock = function(...args) {
      const result = originalUpdateBlock.apply(this, args);
      if (onChange && documentRef.current) {
        onChange({ ...window.wisk.editor.document });
      }
      return result;
    };

    window.wisk.editor.createNewBlock = function(...args) {
      const result = originalCreateNewBlock.apply(this, args);
      if (onChange && documentRef.current) {
        onChange({ ...window.wisk.editor.document });
      }
      return result;
    };

    window.wisk.editor.deleteBlock = function(...args) {
      const result = originalDeleteBlock.apply(this, args);
      if (onChange && documentRef.current) {
        onChange({ ...window.wisk.editor.document });
      }
      return result;
    };
  }, [onChange]);

  // Internal method to set document
  const setDocumentInternal = useCallback((doc: WiskDocument) => {
    if (!window.wisk?.editor) return;

    window.wisk.editor.document = doc;
    documentRef.current = doc;

    // Clear existing editor content
    if (editorRef.current) {
      editorRef.current.innerHTML = '';
    }

    // Render elements
    doc.data.elements.forEach((element, index) => {
      if (editorRef.current && window.wisk?.editor) {
        const elementDiv = document.createElement('div');
        elementDiv.id = `div-${element.id}`;
        elementDiv.classList.add('rndr');

        const blockElement = document.createElement(element.component);
        blockElement.id = element.id;

        // Set element value
        if (blockElement && typeof blockElement.setValue === 'function') {
          blockElement.setValue('', element.value);
        }

        elementDiv.appendChild(blockElement);
        editorRef.current.appendChild(elementDiv);
      }
    });
  }, []);

  // Expose methods via ref
  useImperativeHandle(ref, () => ({
    getDocument: () => {
      return window.wisk?.editor?.document || documentRef.current;
    },
    
    setDocument: (doc: WiskDocument) => {
      setDocumentInternal(doc);
      if (onChange) {
        onChange(doc);
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
  }), [onChange, setDocumentInternal]);

  return (
    <div 
      ref={containerRef}
      className={`wisk-editor-container ${className}`}
      style={style}
    >
      {/* Main editor area */}
      <div className="main">
        <div className="mix">
          <nav id="nav">
            <div className="nav-top-icons">
              <button className="nav-button" id="menu-1">
                <img src="/js/plugins/icons/menu.svg" alt="Menu" className="plugin-icon" />
              </button>
            </div>
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