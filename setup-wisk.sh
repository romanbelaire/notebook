#!/bin/bash

# Wisk Setup Script
# This script helps set up Wisk as a Git submodule dependency

set -e

echo "🚀 Setting up Wisk as a Git submodule..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the project root."
    exit 1
fi

# Check if whisk-submodule already exists
if [ -d "whisk-submodule" ]; then
    echo "⚠️  whisk-submodule already exists. Removing..."
    rm -rf whisk-submodule
fi

# Add Wisk as submodule
echo "📦 Adding Wisk as Git submodule..."
git submodule add https://github.com/sohzm/wisk.git whisk-submodule

# Initialize submodules
echo "🔄 Initializing submodules..."
git submodule update --init --recursive

# Create environment file if it doesn't exist
if [ ! -f ".env.development" ]; then
    echo "📝 Creating .env.development file..."
    cat > .env.development << EOF
# Wisk Editor Configuration
REACT_APP_WISK_PATH=/whisk-submodule
EOF
    echo "✅ Created .env.development"
else
    echo "⚠️  .env.development already exists. Please add REACT_APP_WISK_PATH=/whisk-submodule manually."
fi

# Check if Vite config needs updating
if [ -f "ui/vite.config.ts" ]; then
    if ! grep -q "allow.*\['..'\]" ui/vite.config.ts; then
        echo "⚠️  Please update ui/vite.config.ts to allow serving files from parent directories:"
        echo "   server: { fs: { allow: ['..'] } }"
    else
        echo "✅ Vite config already configured"
    fi
fi

# Create .gitmodules if it doesn't exist
if [ ! -f ".gitmodules" ]; then
    echo "📝 Creating .gitmodules file..."
    cat > .gitmodules << EOF
[submodule "whisk-submodule"]
	path = whisk-submodule
	url = https://github.com/sohzm/wisk.git
EOF
    echo "✅ Created .gitmodules"
fi

echo ""
echo "🎉 Wisk setup complete!"
echo ""
echo "Next steps:"
echo "1. Start your development server: npm run dev"
echo "2. Test the integration by importing WiskTest component"
echo "3. Check that assets load from /whisk-submodule/"
echo ""
echo "To update Wisk in the future:"
echo "  git submodule update --remote whisk-submodule"
echo "  git add whisk-submodule"
echo "  git commit -m 'Update Wisk to latest version'"
echo ""
echo "For more information, see WISK_DEVELOPMENT_SETUP.md" 