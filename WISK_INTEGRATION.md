# Wisk Editor Integration

This document explains the integration of the Wisk block-based editor as a replacement for Quill in the ScratchPad component.

## Overview

Wisk is a modern, block-based document editor built with vanilla JavaScript and Web Components. Unlike traditional rich text editors like Quill, Wisk uses a modular architecture where each content type (text, headings, code blocks, etc.) is implemented as a separate web component.

## Integration Components

### 1. React Wrapper (`ui/src/components/WiskEditor.tsx`)

A React wrapper component that:
- Loads Wisk assets dynamically
- Provides a React-friendly API
- Handles data conversion between HTML and Wisk's JSON format
- Manages the editor lifecycle

### 2. Wisk ScratchPad (`ui/src/components/WiskScratchPad.tsx`)

A new ScratchPad component that:
- Uses WiskEditor instead of Quill
- Maintains the same UI/UX as the original
- Handles drag-and-drop for insights
- Provides auto-save functionality

### 3. CSS Integration (`ui/src/styles/wisk-integration.css`)

Styles that:
- Map Wisk's CSS variables to your design system
- Ensure consistent theming
- Provide responsive design
- Handle dark mode

## Key Benefits

1. **Modular Architecture**: Each content type is a separate component
2. **Extensibility**: Easy to add new block types
3. **Better Performance**: More efficient rendering for complex documents
4. **Rich Content Types**: Built-in support for LaTeX, Mermaid, charts, etc.
5. **Real-time Collaboration**: Built-in sync capabilities
6. **Offline Support**: Works without internet connection

## Data Format Comparison

### Quill (HTML/Delta)
```html
<h1>Hello World</h1>
<p>This is a paragraph with <strong>bold</strong> text.</p>
```

### Wisk (JSON)
```json
{
  "data": {
    "config": {
      "name": "Document",
      "theme": "default", 
      "plugins": ["text-element", "heading1-element"]
    },
    "elements": [
      {
        "id": "heading_1",
        "component": "heading1-element",
        "value": { "textContent": "Hello World" }
      },
      {
        "id": "text_1", 
        "component": "text-element",
        "value": { "textContent": "This is a paragraph with bold text." }
      }
    ]
  }
}
```

## React Wrapper for Open Source Contribution

The React wrapper in `whisk/react-wrapper/` is designed to be contributed back to the original Wisk repository. It includes:

### Package Structure
```
whisk/react-wrapper/
├── package.json          # NPM package configuration
├── tsconfig.json        # TypeScript configuration  
├── rollup.config.js     # Build configuration
├── README.md           # Usage documentation
└── src/
    ├── index.ts        # Main exports
    ├── types.ts        # TypeScript definitions
    ├── utils.ts        # Utility functions
    └── WiskEditor.tsx  # React component
```

### Key Features
- **Framework Agnostic**: Core utilities work with any framework
- **TypeScript Support**: Full type definitions
- **Zero Dependencies**: Only peer dependencies on React
- **Tree Shakeable**: ESM and CJS builds
- **Conversion Utilities**: HTML ↔ Wisk format conversion

### Contributing Back to Wisk

To contribute the React wrapper to the main Wisk repository:

1. **Fork the Wisk repository**
2. **Add the react-wrapper directory** to the root
3. **Update main package.json** to include workspace:
   ```json
   {
     "workspaces": ["react-wrapper"]
   }
   ```
4. **Add build scripts**:
   ```json
   {
     "scripts": {
       "build:react": "cd react-wrapper && npm run build",
       "publish:react": "cd react-wrapper && npm publish"
     }
   }
   ```
5. **Submit a pull request** with:
   - Clear description of the React wrapper
   - Usage examples
   - Documentation updates

## Migration Guide

### From Quill to Wisk

1. **Install dependencies** (none needed - uses whisk assets)
2. **Replace component**:
   ```tsx
   // Old
   import { ScratchPad } from './components/ScratchPad';
   
   // New  
   import { WiskScratchPad } from './components/WiskScratchPad';
   ```
3. **Add CSS imports**:
   ```tsx
   import './styles/wisk-integration.css';
   ```
4. **Update data handling** (automatic conversion included)

### Incremental Migration

You can run both editors side-by-side:
- Keep existing Quill-based ScratchPad for existing notes
- Use Wisk for new notes
- Gradually migrate content using conversion utilities

## Development Setup

### 1. Set up Wisk as a Git Submodule (Recommended)
```bash
# Run the setup script
chmod +x setup-wisk.sh
./setup-wisk.sh

# Or manually:
git submodule add https://github.com/sohzm/wisk.git whisk-submodule
git submodule update --init --recursive
```

### 2. Configure Development Server
The Vite config has been updated to serve files from parent directories:
```javascript
// ui/vite.config.ts
server: {
  fs: {
    allow: ['..'] // Allow serving files from parent directories
  }
}
```

### 3. Environment Configuration
Create `.env.development`:
```bash
REACT_APP_WISK_PATH=/whisk-submodule
```

### 4. Asset Loading
Assets are automatically loaded from the configured path:
```tsx
<WiskEditor 
  // ... other props
  // Assets loaded from REACT_APP_WISK_PATH or default
/>
```

### 5. Custom Asset URL
```tsx
// Override in environment variables
REACT_APP_WISK_PATH=https://cdn.wisk.cc/v1

// Or programmatically
import { getWiskAssetPath } from './config/wisk';
console.log('Current path:', getWiskAssetPath());
```

## Advanced Usage

### Custom Plugins
```tsx
<WiskEditor
  plugins={['latex-element', 'mermaid-element', 'chart-element']}
  onChange={handleChange}
/>
```

### Theming
```css
.wisk-editor-container {
  --bg-1: #ffffff;
  --fg-1: #000000;
  --accent: #007acc;
}
```

### API Access
```tsx
const editorRef = useRef<WiskEditorRef>(null);

// Add a code block
editorRef.current?.addBlock('code-element', {
  language: 'javascript',
  code: 'console.log("Hello");'
});

// Convert to HTML
const html = editorRef.current?.convertToHtml();
```

## Troubleshooting

### Assets Not Loading
- Ensure whisk-submodule is initialized: `git submodule update --init --recursive`
- Check browser console for 404 errors
- Verify the asset path: `console.log(getWiskAssetPath())`
- Ensure Vite config allows parent directory access
- Verify CORS settings if loading from CDN

### Styling Issues
- Import `wisk-integration.css`
- Check CSS variable mappings
- Ensure proper z-index for overlays

### TypeScript Errors
- Install `@types/react` and `@types/react-dom`
- Check tsconfig.json compatibility
- Use type-only imports for interfaces

## Future Enhancements

1. **Plugin Marketplace**: Add support for custom plugins
2. **Advanced Sync**: Real-time collaboration features
3. **Export Formats**: PDF, DOCX, Markdown export
4. **AI Integration**: Smart writing assistance
5. **Templates**: Document templates and snippets

## Performance Considerations

- Wisk assets are loaded once and cached
- Components are lazy-loaded as needed
- Large documents use virtual scrolling
- Change detection uses efficient diffing

## Browser Support

- Chrome/Edge 88+
- Firefox 85+
- Safari 14+
- Mobile browsers with ES2018 support

---

This integration provides a modern, extensible editor while maintaining backward compatibility and a smooth migration path. 