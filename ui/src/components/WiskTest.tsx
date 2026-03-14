import { useRef, useState } from 'react';
import WiskEditor, { type WiskEditorRef, type WiskDocument } from './WiskEditor';
import { getWiskAssetPath, WISK_CONFIG } from '../config/wisk';

export default function WiskTest() {
  const editorRef = useRef<WiskEditorRef>(null);
  const [document, setDocument] = useState<WiskDocument | null>(null);
  const [html, setHtml] = useState('');

  const handleChange = (doc: WiskDocument) => {
    setDocument(doc);
    console.log('Document changed:', doc);
  };

  const addHeading = () => {
    if (editorRef.current) {
      editorRef.current.addBlock('heading1-element', { textContent: 'New Heading' });
    }
  };

  const addParagraph = () => {
    if (editorRef.current) {
      editorRef.current.addBlock('text-element', { textContent: 'New paragraph text...' });
    }
  };

  const convertToHtml = () => {
    if (editorRef.current) {
      const htmlContent = editorRef.current.convertToHtml();
      setHtml(htmlContent);
    }
  };

  const loadFromHtml = () => {
    if (editorRef.current) {
      const sampleHtml = '<h1>Sample Heading</h1><p>Sample paragraph with <strong>bold</strong> text.</p>';
      const wiskDoc = editorRef.current.convertFromHtml(sampleHtml);
      editorRef.current.setDocument(wiskDoc);
      setDocument(wiskDoc);
    }
  };

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">Wisk Editor Test</h1>
      
      {/* Configuration Info */}
      <div className="bg-blue-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Configuration:</h3>
        <div className="text-sm space-y-1">
          <div><strong>Asset Path:</strong> {getWiskAssetPath()}</div>
          <div><strong>Environment:</strong> {process.env.NODE_ENV}</div>
          <div><strong>Custom Path:</strong> {process.env.REACT_APP_WISK_PATH || 'Not set'}</div>
        </div>
      </div>
      
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={addHeading}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Add Heading
        </button>
        <button
          onClick={addParagraph}
          className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Add Paragraph
        </button>
        <button
          onClick={convertToHtml}
          className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600"
        >
          Convert to HTML
        </button>
        <button
          onClick={loadFromHtml}
          className="px-3 py-1 bg-orange-500 text-white rounded hover:bg-orange-600"
        >
          Load Sample HTML
        </button>
        <button
          onClick={() => editorRef.current?.focus()}
          className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Focus Editor
        </button>
      </div>

      <div className="border rounded-lg overflow-hidden" style={{ height: '400px' }}>
        <WiskEditor
          ref={editorRef}
          onChange={handleChange}
          placeholder="Start writing or press '/' for commands..."
          className="h-full"
        />
      </div>

      {document && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Document JSON:</h3>
          <pre className="bg-gray-100 p-3 rounded text-xs overflow-auto max-h-32">
            {JSON.stringify(document, null, 2)}
          </pre>
        </div>
      )}

      {html && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Generated HTML:</h3>
          <pre className="bg-gray-100 p-3 rounded text-xs overflow-auto max-h-32">
            {html}
          </pre>
          <div className="mt-2 p-3 border rounded">
            <h4 className="text-sm font-medium mb-1">Rendered HTML:</h4>
            <div dangerouslySetInnerHTML={{ __html: html }} />
          </div>
        </div>
      )}
    </div>
  );
} 