/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        heading: ["Bodoni Moda", "serif"],
        body: ["Plus Jakarta Sans", "sans-serif"]
      },
      colors: {
        gridmind: {
          obsidian: "#080a0f",
          midnight: "#101827",
          carbon: "#1a2333",
          haze: "#8fa5c4",
          pearl: "#f6f2e8",
          gold: "#d7b36f",
          goldDeep: "#9f7c3c",
          emerald: "#3ca67f",
          amber: "#d59750",
          ruby: "#d66f61"
        }
      },
      boxShadow: {
        panel: "0 18px 40px rgba(3, 8, 20, 0.45)",
        glow: "0 0 0 1px rgba(215,179,111,0.3), 0 10px 34px rgba(215,179,111,0.18)"
      }
    }
  },
  plugins: []
};
