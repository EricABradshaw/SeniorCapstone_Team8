import React from 'react'
import './index.scss'

// import components

function App() {
  return (
    <div className="App">
      <div id="app-header">
        <h1>Hello Capstone</h1>
      </div>
      <div id="mainSection">
        <div className="filler"></div>
        <div id="secretImageSection">
          <h1>Secret Image</h1>
        </div>
        <div id="coverImageSection">
          <h1>Cover Image</h1>
        </div>
        <div id="stegoImageSection">
          <h1>Stego Image?</h1>
        </div>
        <div className="filler"></div>
      </div>
      <div id="imageLibrary">
        <div>
          <h3>Image Library</h3>
        </div>
      </div>
    </div>
  )
}

export default App
