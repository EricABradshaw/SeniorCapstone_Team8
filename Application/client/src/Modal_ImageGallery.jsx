import React, { useState } from 'react'
import { GridGallery, getImages } from './GridGallery';

const handleUploadClick = async () => {
  try {
    const selectedFile = await selectFile();
    if (selectedFile) {
      // setStegoImage(selectedFile);
    }
  } catch (error) {
    console.error('Error selecting file:', error);
  }
};

const selectFile = () => {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';

    input.addEventListener('change', (event) => {
      const selectedFile = event.target.files[0];
      resolve(selectedFile);
    });
    input.click();
  });
};

const Modal = ({ isOpen, onClose, selectedItem, handleSecretImageSelect, handleCoverImageSelect }) => {
  const [refreshKey, setRefreshKey] = useState(0);
  const handleRefreshClick = () => {
    setRefreshKey((prevKey) => prevKey + 1)
  }
  // Callback function to be passed to GridGallery
  const handleImageSelect = (index, image) => {
    if (selectedItem === "coverImage")
      handleCoverImageSelect(image);
    if (selectedItem === "secretImage")
      handleSecretImageSelect(image);

    onClose();
  };

  return (
    <div key={refreshKey} className={`modal ${isOpen ? 'open' : ''}`} onClick={onClose}>
      <div className='overlay'></div>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div id="imageGallery">
          <GridGallery onSelect={handleImageSelect} />
        </div>
        <div style={{ marginTop: '20px'}}>
          <button onClick={handleRefreshClick}>Refresh</button>
          <button onClick={onClose} style={{ marginInlineStart: '10px', marginInlineEnd: '10px'}}>Close</button>
          <button onClick={handleUploadClick}>Upload your own</button>
        </div>
      </div>
    </div>
  );
};

export default Modal;
