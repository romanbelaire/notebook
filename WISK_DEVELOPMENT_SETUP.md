# Wisk Development Setup

This guide explains how to set up Wisk as a dependency in your project without duplicating code.

## Option 1: Git Submodule (Recommended)

### Setup
```bash
# Add Wisk as a submodule
git submodule add https://github.com/sohzm/wisk.git whisk-submodule

# Initialize and update submodules
git submodule update --init --recursive
```

### Benefits
- ✅ **Version Control**: Track specific Wisk commits
- ✅ **No Code Duplication**: Wisk stays in its own repo
- ✅ **Easy Updates**: `git submodule update --remote`
- ✅ **Clean History**: Submodule changes are separate

### Usage
```bash
# Update to latest Wisk version
git submodule update --remote whisk-submodule

# Switch to specific Wisk version
cd whisk-submodule
git checkout <commit-hash>
cd ..

# Commit the submodule update
git add whisk-submodule
git commit -m "Update Wisk to latest version"
```

## Option 2: NPM Package (Future)

When the React wrapper is published to NPM:

```bash
npm install @wisk/react
```

```tsx
import { WiskEditor } from '@wisk/react';
```

## Option 3: Direct Git Dependency

Add to `package.json`:
```json
{
  "dependencies": {
    "wisk": "github:sohzm/wisk#main"
  }
}
```

## Development Workflow

### 1. Initial Setup
```bash
# Clone your project with submodules
git clone --recursive <your-repo-url>

# Or if already cloned
git submodule update --init --recursive
```

### 2. Development Server Configuration

#### Vite/Webpack Dev Server
```javascript
// vite.config.js or webpack.config.js
export default {
  server: {
    fs: {
      allow: ['..'] // Allow serving files from parent directories
    }
  }
}
```

#### Express/Static Server
```javascript
// Serve whisk-submodule as static files
app.use('/whisk-submodule', express.static('whisk-submodule'));
```

### 3. Environment Configuration

Create `.env` files for different environments:

```bash
# .env.development
REACT_APP_WISK_PATH=/whisk-submodule

# .env.production  
REACT_APP_WISK_PATH=https://cdn.wisk.cc/v1
```

### 4. Asset Loading

The WiskEditor automatically loads assets from the configured path:

```tsx
import { WiskEditor } from './components/WiskEditor';

// Assets loaded from REACT_APP_WISK_PATH or default
<WiskEditor onChange={handleChange} />
```

## File Structure

```
your-project/
├── whisk-submodule/          # Git submodule
│   ├── js/
│   ├── css/
│   ├── style.css
│   └── ...
├── ui/
│   ├── src/
│   │   ├── components/
│   │   │   ├── WiskEditor.tsx
│   │   │   └── WiskScratchPad.tsx
│   │   ├── config/
│   │   │   └── wisk.ts       # Configuration
│   │   └── styles/
│   │       └── wisk-integration.css
│   └── public/
└── package.json
```

## Configuration Options

### Asset Paths
```typescript
// ui/src/config/wisk.ts
export const WISK_CONFIG = {
  PATHS: {
    SUBMODULE: '/whisk-submodule',    // Git submodule
    LOCAL: '/whisk',                  // Local copy
    CDN: 'https://cdn.wisk.cc/v1',    // CDN
    CUSTOM: process.env.REACT_APP_WISK_PATH
  }
};
```

### Environment Variables
```bash
# Development
REACT_APP_WISK_PATH=/whisk-submodule

# Production
REACT_APP_WISK_PATH=https://cdn.wisk.cc/v1

# Custom local path
REACT_APP_WISK_PATH=/custom/whisk/path
```

## Troubleshooting

### Assets Not Loading
```bash
# Check if submodule is initialized
git submodule status

# Reinitialize if needed
git submodule update --init --recursive

# Check file permissions
ls -la whisk-submodule/
```

### Path Issues
```javascript
// Verify asset path in browser console
console.log('Wisk path:', getWiskAssetPath());

// Check network tab for 404 errors
// Ensure dev server serves the correct path
```

### TypeScript Errors
```bash
# If using the React wrapper package
npm install @wisk/react

# Or add types manually
declare module 'wisk' {
  // Type definitions
}
```

## Deployment

### Static Hosting (Vercel, Netlify)
```bash
# Build includes submodule files
npm run build

# Deploy with submodule
git push --recurse-submodules=on-demand
```

### Docker
```dockerfile
# Copy submodule files
COPY whisk-submodule /app/public/whisk-submodule

# Or use multi-stage build
FROM node:18 as builder
COPY whisk-submodule /app/whisk-submodule
# ... rest of build
```

### CDN Setup
```bash
# Upload whisk-submodule to CDN
aws s3 sync whisk-submodule s3://your-cdn/whisk/

# Update environment variable
REACT_APP_WISK_PATH=https://your-cdn.com/whisk/
```

## Best Practices

1. **Pin Versions**: Use specific commit hashes for stability
2. **Document Changes**: Keep track of Wisk updates
3. **Test Thoroughly**: Verify functionality after updates
4. **Backup Strategy**: Keep local copy for critical deployments
5. **CI/CD**: Include submodule updates in build pipeline

## Migration from Local Copy

If you currently have a local copy of Wisk:

```bash
# 1. Remove local copy
rm -rf whisk/

# 2. Add as submodule
git submodule add https://github.com/sohzm/wisk.git whisk-submodule

# 3. Update import paths
# Change /whisk to /whisk-submodule in your code

# 4. Test thoroughly
npm run dev
```

## Contributing Back

When contributing the React wrapper to Wisk:

```bash
# 1. Fork Wisk repository
# 2. Add react-wrapper directory
# 3. Submit pull request
# 4. Update your submodule when merged
git submodule update --remote whisk-submodule
```

This setup provides a clean, maintainable way to use Wisk as a dependency while keeping your codebase lean and up-to-date. 