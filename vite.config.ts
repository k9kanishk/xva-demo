import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/xva-demo/',   // <-- exact repo name, with leading & trailing slash
})
