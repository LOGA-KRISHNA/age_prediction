import { useState } from 'react'
import './App.css'
import AgePredictionApp from './components/AgePredictionApp'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <AgePredictionApp />
    </>
  )
}

export default App
