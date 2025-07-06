import { createRoot } from 'react-dom/client'
import * as ReactDOM from 'react-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.tsx'
import { ToastProvider } from './components/ToastProvider'

const queryClient = new QueryClient()

const root = createRoot(document.getElementById('root')!);

// Poly-fill findDOMNode *after* createRoot (React 18 disables it during concurrent roots)
if (!(ReactDOM as any).findDOMNode) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (ReactDOM as any).findDOMNode = (inst: any) =>
    inst && typeof inst === 'object' && 'current' in inst ? inst.current : inst;
}

// Ensure polyfill is present on possible nested default export used by some bundlers
if ((ReactDOM as any).default && !(ReactDOM as any).default.findDOMNode) {
  (ReactDOM as any).default.findDOMNode = (ReactDOM as any).findDOMNode;
}

root.render(
  <QueryClientProvider client={queryClient}>
    <ToastProvider>
      <App />
    </ToastProvider>
  </QueryClientProvider>
)
