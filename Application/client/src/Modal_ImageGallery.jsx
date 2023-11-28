import React from 'react'
import { GridGallery } from './GridGallery';

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
      </div>
    </div>
  );
};

export default Modal;
