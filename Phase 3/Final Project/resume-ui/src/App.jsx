import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import ResumeInputForm from './components/ResumeInputForm'
import PredictionAndWordCloud from './components/PredictionAndWordCloud'
import RewriteSuggestions from './components/RewriteSuggestions'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ResumeInputForm />} />
        <Route path="/analysis" element={<PredictionAndWordCloud />} />
        <Route path="/rewrite" element={<RewriteSuggestions />} />
      </Routes>
    </Router>
  )
}

export default App
