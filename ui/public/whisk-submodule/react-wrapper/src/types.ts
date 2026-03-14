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
  /** Initial document content */
  document?: WiskDocument;
  /** Called when document changes */
  onChange?: (document: WiskDocument) => void;
  /** Whether editor is read-only */
  readonly?: boolean;
  /** Custom theme */
  theme?: string;
  /** Additional plugins to load */
  plugins?: string[];
  /** Custom CSS styles */
  className?: string;
  /** Custom inline styles */
  style?: Record<string, any>;
  /** Backend URL for sync features */
  backendUrl?: string;
  /** Enable AI features */
  enableAI?: boolean;
  /** Custom placeholder text */
  placeholder?: string;
}

export interface WiskEditorRef {
  /** Get current document */
  getDocument: () => WiskDocument | null;
  /** Set document content */
  setDocument: (doc: WiskDocument) => void;
  /** Focus the editor */
  focus: () => void;
  /** Add a new block */
  addBlock: (type: string, value?: any, afterElementId?: string) => string;
  /** Delete a block */
  deleteBlock: (elementId: string) => void;
  /** Update a block */
  updateBlock: (elementId: string, value: any) => void;
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