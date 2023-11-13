import React from 'react';
import ReactDOM from 'react-dom';
import {BrowserRouter, Routes, Route} from 'react-router-dom'
import './index.scss';
import Generate from './Generate'
import Home from './Home'

export default function App(){
  return(
    <BrowserRouter>
      <h1>Test</h1>
      <Routes>
        <Route path='/' element={<Home />}></Route>
        <Route path='/Generate' element={<Generate />}></Route>
      </Routes>
    </BrowserRouter>

  )
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App/>)
