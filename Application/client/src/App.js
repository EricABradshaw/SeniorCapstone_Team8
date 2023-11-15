import React, { useState,useRef } from 'react'
import './index.scss'

// import components
import Gallery from './GridGallery'

function App() {

  const coverImageRef = useRef(null);
  const [selectedImage, setSelectedImage] = useState(null);

  // Callback function to be passed to GridGallery
  const handleImageSelect = (index, image) => {
    console.log('Selected Image:', image);
    setSelectedImage(image);
  };

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
        <div id="coverImageSection" ref={coverImageRef}>
        {selectedImage ? (
          <img
            src={selectedImage.src}
            alt={selectedImage.alt || ''}
            width={selectedImage.width}
            height={selectedImage.height}
          />
        ) : (
          <h1>Cover Image</h1>
        )}
        </div>
        <div id="stegoImageSection">
          <h1>Stego Image?</h1>
        </div>
        <div className="filler"></div>
      </div>
      <div id="imageLibrary">
        <div>
          <h3>Image Library</h3>
          <div id="imageGallery">
            <Gallery onSelect={handleImageSelect}/>
          </div>
        </div>
      </div>
    </div>
  )
}
export default App
