import { createRoot } from 'react-dom/client'
import * as ReactDOM from 'react-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.tsx'
import { ToastProvider } from './components/ToastProvider'

const queryClient = new QueryClient()

const root = createRoot(document.getElementById('root')!);

// React 18 still exposes `findDOMNode`; if it is unavailable (e.g.
// future React versions), third-party libraries that rely on it will break
// at runtime. We no longer attempt to monkey-patch the import namespace because
// ESBuild treats namespace objects as immutable. If a library strictly requires
// `findDOMNode`, switch to a legacy root or upgrade the library.

root.render(
  <QueryClientProvider client={queryClient}>
    <ToastProvider>
      <App />
    </ToastProvider>
  </QueryClientProvider>
)
