// Wisk Editor Configuration

// Browser-compatible environment detection
const isDevelopment = typeof process !== 'undefined' 
  ? process.env.NODE_ENV === 'development'
  : window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

export const WISK_CONFIG = {
  // Asset paths - can be configured for different environments
  ASSETS_PATH: isDevelopment
    ? '/whisk-submodule'  // Local development with submodule
    : '/whisk',           // Production (could be CDN)
  
  // Alternative paths for different setups
  PATHS: {
    SUBMODULE: '/whisk-submodule',
    LOCAL: '/whisk',
    CDN: 'https://cdn.wisk.cc/v1',
    CUSTOM: (typeof process !== 'undefined' ? process.env.REACT_APP_WISK_PATH : null) || '/whisk-submodule'
  },
  
  // Editor settings
  DEFAULT_THEME: 'default',
  DEFAULT_PLUGINS: [
    'text-element',
    'heading1-element', 
    'heading2-element',
    'heading3-element',
    'code-element',
    'list-element',
    'numbered-list-element',
    'quote-element',
    'image-element'
  ],
  
  // Performance settings
  CHANGE_POLLING_INTERVAL: 500, // ms
  ASSET_LOAD_TIMEOUT: 5000,     // ms
  MAX_INIT_ATTEMPTS: 50,
  
  // Feature flags
  ENABLE_AI: false,
  ENABLE_SYNC: false,
  ENABLE_REAL_TIME: false,
} as const;

// Helper function to get the current asset path
export function getWiskAssetPath(): string {
  return WISK_CONFIG.PATHS.CUSTOM;
}

// Helper function to get full asset URL
export function getWiskAssetUrl(asset: string): string {
  return `${getWiskAssetPath()}${asset}`;
} 