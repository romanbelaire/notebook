/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Open Sans", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        primaryBg: '#142338',
        secondaryBg: '#16263c',
        headerBg: '#111c2d',
        trim: '#182529aa',
        default: '#cadde2',
        defaultText: '#cadde2',
        light: '#ffffff',
        'light-text': '#ffffff',
        accentText: '#d48e33',
        buttonBg: '#40404f',
        'chat-assistant-bg': '#353f50',
      },
      keyframes: {
        wiggle: {
          '0%, 100%': { transform: 'rotate(-10deg)' },
          '50%': { transform: 'rotate(10deg)' },
        },
      },
      animation: {
        wiggle: 'wiggle 0.5s ease-in-out',
      },
      boxShadow: {
        'inner-strong': 'inset 0 0 8px 3px rgba(0,0,0,0.8)',
      },
    },
  },
  safelist: [
    'animate-wiggle',
    'shadow-[0_0_10px_rgba(255,255,255,0.2)]',
    'drop-shadow-[0_0_6px_rgba(255,255,255,0.8)]',
  ],
  plugins: [],
}; 