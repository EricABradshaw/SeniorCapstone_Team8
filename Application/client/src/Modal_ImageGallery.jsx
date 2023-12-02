import React from 'react'
import { GridGallery } from './GridGallery';

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

  // Callback function to be passed to GridGallery
  const handleImageSelect = (index, image) => {
    if (selectedItem === "coverImage")
      handleCoverImageSelect(image);
    if (selectedItem === "secretImage")
      handleSecretImageSelect(image);

    onClose();
  };

  return (
    <div className={`modal ${isOpen ? 'open' : ''}`} onClick={onClose}>
      <div className='overlay'></div>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div id="imageGallery">
          <GridGallery onSelect={handleImageSelect} />
        </div>
        <button onClick={onClose}>Close</button>
        <button onClick={handleUploadClick}>Upload your own</button>
      </div>
    </div>
  );
};

export default Modal;
