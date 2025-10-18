import { useState } from 'react'

export default function App() {
  const [count, setCount] = useState(0)
  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 24 }}>
      <h1>XVA Demo</h1>
      <p>If you see this, GitHub Pages build is working.</p>
      <button onClick={() => setCount(c => c + 1)}>
        Clicks: {count}
      </button>
    </div>
  )
}
