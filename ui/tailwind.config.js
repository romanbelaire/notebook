const ch = (v) => `rgb(from var(${v}) r g b / <alpha-value>)`;

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
        primaryBg: ch("--color-primary-bg"),
        secondaryBg: ch("--color-secondary-bg"),
        headerBg: ch("--color-header-bg"),
        trim: ch("--color-trim"),
        default: ch("--color-defaultText"),
        defaultText: ch("--color-defaultText"),
        light: ch("--color-light-text"),
        "light-text": ch("--color-light-text"),
        accentText: ch("--color-accent-text"),
        buttonBg: ch("--color-button-bg"),
        "chat-assistant-bg": ch("--color-chat-assistant-bg"),
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