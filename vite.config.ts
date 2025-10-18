import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// IMPORTANT: replace REPO-NAME with your exact GitHub repo name
export default defineConfig({
  plugins: [react()],
  base: '/xva-demo/',   // <-- include leading and trailing slashes
})
