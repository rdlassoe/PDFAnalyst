import type { Config } from "tailwindcss";

// Paleta sobria: blanco cálido, gris carbón, azul petróleo como color de acción.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#FAF9F6",
        panel: "#F3F1EC",
        line: "#E2DFD8",
        ink: { DEFAULT: "#2B2D2F", soft: "#5B5F63", faint: "#8A8E92" },
        petrol: { DEFAULT: "#0F4C5C", dark: "#0A3844", tint: "#E3EEF0" },
        ok: "#2F6B4F",
        warn: "#8A5A00",
        bad: "#9B2C2C",
      },
      keyframes: {
        "toast-in": { from: { opacity: "0", transform: "translateX(24px) scale(0.98)" }, to: { opacity: "1", transform: "none" } },
        "toast-out": { from: { opacity: "1", transform: "none" }, to: { opacity: "0", transform: "translateX(24px) scale(0.98)" } },
        "toast-progress": { from: { transform: "scaleX(1)" }, to: { transform: "scaleX(0)" } },
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "pop-in": { from: { opacity: "0", transform: "translateY(8px) scale(0.97)" }, to: { opacity: "1", transform: "none" } },
        "slide-in-right": { from: { transform: "translateX(100%)" }, to: { transform: "none" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: {
        "toast-in": "toast-in 220ms cubic-bezier(0.21, 1.02, 0.73, 1) both",
        "toast-out": "toast-out 180ms ease-in both",
        "toast-progress": "toast-progress linear forwards",
        "fade-in": "fade-in 150ms ease-out both",
        "pop-in": "pop-in 180ms cubic-bezier(0.21, 1.02, 0.73, 1) both",
        "slide-in-right": "slide-in-right 220ms cubic-bezier(0.21, 1.02, 0.73, 1) both",
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
