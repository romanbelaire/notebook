/**
 * Load Wisk CSS and JS assets dynamically
 */
export async function loadWiskAssets(baseUrl = ''): Promise<void> {
  const assetsToLoad = [
    // CSS files
    { type: 'css', url: `${baseUrl}/css/mini-dialog.css` },
    { type: 'css', url: `${baseUrl}/css/right-sidebar.css` },
    { type: 'css', url: `${baseUrl}/css/left-sidebar.css` },
    { type: 'css', url: `${baseUrl}/style.css` },
    { type: 'css', url: `${baseUrl}/global.css` },
    { type: 'css', url: `${baseUrl}/js/theme/variables.css` },
    
    // JS files
    { type: 'js', url: `${baseUrl}/global.js` },
    { type: 'js', url: `${baseUrl}/js/wisk.js` },
    { type: 'js', url: `${baseUrl}/js/utils.js` },
    { type: 'js', url: `${baseUrl}/js/polyfill.js` },
    { type: 'js', url: `${baseUrl}/js/storage/storage.js` },
    { type: 'js', url: `${baseUrl}/js/sync/sync.js` },
    { type: 'js', url: `${baseUrl}/js/plugins/plugins.js` },
    { type: 'js', url: `${baseUrl}/js/editor.js` },
    { type: 'js', url: `${baseUrl}/js/mini-dialog.js` },
    { type: 'js', url: `${baseUrl}/js/right-sidebar.js` },
    { type: 'js', url: `${baseUrl}/js/left-sidebar.js` },
    { type: 'js', url: `${baseUrl}/js/paste-handler.js` },
  ];

  const loadPromises = assetsToLoad.map(asset => {
    return new Promise<void>((resolve, reject) => {
      if (asset.type === 'css') {
        // Check if already loaded
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
        // Check if already loaded
        if (document.querySelector(`script[src="${asset.url}"]`)) {
          resolve();
          return;
        }
        
        const script = document.createElement('script');
        script.src = asset.url;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load JS: ${asset.url}`));
        document.head.appendChild(script);
      }
    });
  });

  await Promise.all(loadPromises);
}

/**
 * Initialize the Wisk editor with configuration
 */
export async function initializeWisk(config: {
  readonly?: boolean;
  backendUrl?: string;
  enableAI?: boolean;
}): Promise<void> {
  // Wait for wisk object to be available
  let attempts = 0;
  const maxAttempts = 50;
  
  while (!window.wisk && attempts < maxAttempts) {
    await new Promise(resolve => setTimeout(resolve, 100));
    attempts++;
  }
  
  if (!window.wisk) {
    throw new Error('Wisk failed to initialize after loading assets');
  }

  // Configure wisk
  window.wisk.editor.readonly = config.readonly || false;
  window.wisk.editor.backendUrl = config.backendUrl || '';
  
  // Initialize empty document if none exists
  if (!window.wisk.editor.document) {
    window.wisk.editor.document = {
      data: {
        config: {
          name: 'Untitled Document',
          theme: 'default',
          plugins: window.wisk.plugins.defaultPlugins || [],
        },
        elements: [],
        pluginData: {},
      }
    };
  }

  // Load default plugins
  if (window.wisk.plugins && window.wisk.plugins.defaultPlugins) {
    for (const plugin of window.wisk.plugins.defaultPlugins) {
      try {
        await window.wisk.plugins.loadPlugin(plugin);
      } catch (error) {
        console.warn(`Failed to load plugin ${plugin}:`, error);
      }
    }
  }
}

/**
 * Convert HTML content to Wisk document format
 */
export function htmlToWiskDocument(html: string): any {
  // Basic conversion - can be enhanced
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const elements: any[] = [];

  // Convert paragraphs and basic elements
  const textNodes = doc.querySelectorAll('p, h1, h2, h3, h4, h5, h6, div');
  
  textNodes.forEach((node, index) => {
    let component = 'text-element';
    
    // Determine component type based on tag
    if (node.tagName === 'H1') component = 'heading1-element';
    else if (node.tagName === 'H2') component = 'heading2-element';
    else if (node.tagName === 'H3') component = 'heading3-element';
    else if (node.tagName === 'H4') component = 'heading4-element';
    else if (node.tagName === 'H5') component = 'heading5-element';
    else if (node.tagName === 'H6') component = 'heading5-element'; // H6 maps to H5 in Wisk
    
    elements.push({
      id: `element_${index}`,
      component,
      value: {
        textContent: node.textContent || '',
        html: node.innerHTML || '',
      }
    });
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

/**
 * Convert Wisk document to HTML
 */
export function wiskDocumentToHtml(doc: any): string {
  if (!doc?.data?.elements) return '';
  
  let html = '';
  
  doc.data.elements.forEach((element: any) => {
    const value = element.value || {};
    
    switch (element.component) {
      case 'heading1-element':
        html += `<h1>${value.textContent || value.html || ''}</h1>`;
        break;
      case 'heading2-element':
        html += `<h2>${value.textContent || value.html || ''}</h2>`;
        break;
      case 'heading3-element':
        html += `<h3>${value.textContent || value.html || ''}</h3>`;
        break;
      case 'heading4-element':
        html += `<h4>${value.textContent || value.html || ''}</h4>`;
        break;
      case 'heading5-element':
        html += `<h5>${value.textContent || value.html || ''}</h5>`;
        break;
      case 'text-element':
      default:
        html += `<p>${value.textContent || value.html || ''}</p>`;
        break;
    }
  });
  
  return html;
} 