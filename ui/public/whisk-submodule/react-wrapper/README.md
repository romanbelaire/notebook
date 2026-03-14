# @wisk/react

React wrapper for the [Wisk](https://wisk.cc) block-based document editor.

## Installation

```bash
npm install @wisk/react
```

## Basic Usage

```tsx
import React, { useRef } from 'react';
import { WiskEditor, WiskEditorRef, WiskDocument } from '@wisk/react';

function MyApp() {
  const editorRef = useRef<WiskEditorRef>(null);
  
  const handleChange = (document: WiskDocument) => {
    console.log('Document changed:', document);
  };

  const addHeading = () => {
    if (editorRef.current) {
      editorRef.current.addBlock('heading1-element', { textContent: 'New Heading' });
    }
  };

  return (
    <div>
      <button onClick={addHeading}>Add Heading</button>
      <WiskEditor
        ref={editorRef}
        onChange={handleChange}
        placeholder="Start writing..."
        className="my-editor"
        style={{ height: '500px' }}
      />
    </div>
  );
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `document` | `WiskDocument` | undefined | Initial document content |
| `onChange` | `(doc: WiskDocument) => void` | undefined | Called when document changes |
| `readonly` | `boolean` | `false` | Whether editor is read-only |
| `theme` | `string` | `'default'` | Editor theme |
| `plugins` | `string[]` | `[]` | Additional plugins to load |
| `className` | `string` | `''` | CSS class for container |
| `style` | `Record<string, any>` | `{}` | Inline styles for container |
| `backendUrl` | `string` | `''` | Backend URL for sync features |
| `enableAI` | `boolean` | `false` | Enable AI features |
| `placeholder` | `string` | `'Write something or press \'/\' for commands'` | Placeholder text |

## Ref Methods

The `WiskEditorRef` provides these methods:

- `getDocument()` - Get current document
- `setDocument(doc)` - Set document content
- `focus()` - Focus the editor
- `addBlock(type, value?, afterElementId?)` - Add a new block
- `deleteBlock(elementId)` - Delete a block
- `updateBlock(elementId, value)` - Update a block

## Document Format

The Wisk document format is JSON-based:

```typescript
interface WiskDocument {
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

interface WiskElement {
  id: string;
  component: string;
  value: any;
}
```

## Available Block Types

- `text-element` - Regular text/paragraph
- `heading1-element` through `heading5-element` - Headings
- `code-element` - Code blocks
- `image-element` - Images
- `list-element` - Bullet lists
- `numbered-list-element` - Numbered lists
- `quote-element` - Blockquotes
- `table-element` - Tables
- `latex-element` - LaTeX math
- `mermaid-element` - Mermaid diagrams
- `chart-element` - Charts
- And many more...

## Conversion Utilities

Convert between HTML and Wisk format:

```tsx
import { htmlToWiskDocument, wiskDocumentToHtml } from '@wisk/react';

// Convert HTML to Wisk document
const wiskDoc = htmlToWiskDocument('<h1>Hello</h1><p>World</p>');

// Convert Wisk document to HTML
const html = wiskDocumentToHtml(wiskDoc);
```

## Styling

The editor uses CSS variables for theming. You can customize the appearance by overriding these variables:

```css
.wisk-editor-container {
  --bg-1: #ffffff;
  --bg-2: #f8f9fa;
  --fg-1: #212529;
  --fg-2: #6c757d;
  --border-1: #dee2e6;
  /* ... and many more */
}
```

## Integration with Existing Apps

To integrate with an existing React app that uses a different rich text editor:

1. Convert existing content to Wisk format using `htmlToWiskDocument()`
2. Use the `WiskEditor` component
3. Convert back to your format using `wiskDocumentToHtml()` when needed

## Development

This package is designed to be framework-agnostic and can be contributed back to the main Wisk repository.

## License

Licensed under the Functional Source License (FSL), Version 1.1, with Apache License Version 2.0 as the Future License.

## Contributing

Contributions are welcome! Please submit pull requests to the main [Wisk repository](https://github.com/sohzm/wisk). 