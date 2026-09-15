/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        canvas: {
          DEFAULT: '#090A0F',
          subtle: '#0E1017',
          surface: '#12151E',
          elevated: '#171B26',
          hover: '#1E2332',
        },
        border: {
          subtle: 'rgba(255, 255, 255, 0.07)',
          DEFAULT: 'rgba(255, 255, 255, 0.12)',
          strong: 'rgba(255, 255, 255, 0.22)',
        },
        apple: {
          blue: '#0071E3',
          'blue-hover': '#0077ED',
          'blue-subtle': 'rgba(0, 113, 227, 0.12)',
        },
        decision: {
          auto: '#10B981',
          'auto-bg': 'rgba(16, 185, 129, 0.07)',
          'auto-border': 'rgba(16, 185, 129, 0.25)',
          escalate: '#EF4444',
          'escalate-amber': '#F59E0B',
          'escalate-bg': 'rgba(239, 68, 68, 0.07)',
          'escalate-border': 'rgba(239, 68, 68, 0.25)',
        },
      },
      boxShadow: {
        'subtle-card': '0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 0 1px 1px rgba(255, 255, 255, 0.08)',
        'hero-auto': '0 12px 36px -4px rgba(16, 185, 129, 0.15), 0 0 1px 1px rgba(16, 185, 129, 0.3)',
        'hero-escalate': '0 12px 36px -4px rgba(239, 68, 68, 0.15), 0 0 1px 1px rgba(239, 68, 68, 0.3)',
        'tactile-btn': '0 2px 8px rgba(0, 113, 227, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.35s ease-out forwards',
        'slide-up': 'slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
