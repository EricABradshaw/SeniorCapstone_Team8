import React, { useState } from 'react'
import { GridGallery } from './GridGallery';
import { Button } from 'react-bootstrap'

const Modal = ({ isOpen, onClose, selectedItem, handleSecretImageSelect, handleCoverImageSelect }) => {
  const [refreshKey, setRefreshKey] = useState(0);
  const handleRefreshClick = () => {
    setRefreshKey((prevKey) => prevKey + 1)
  }

 const handleUploadClick = async () => {
    try {
      const selectedFile = await selectFile();
      if (selectedFile) {
        let base64String = await fileToBase64(selectedFile, 224, 224);
        const img = {
          src: base64String
        }
        
        handleImageSelect(0, img);
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

  const fileToBase64 = (file, maxWidth, maxHeight) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const img = new Image();
        img.src = reader.result;
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = maxWidth;
          canvas.height = maxHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, maxWidth, maxHeight);
          resolve(canvas.toDataURL('image/jpeg')); 
        };
      };
      reader.onerror = (error) => reject(error);
    });
  };

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
      <div className="modal-content bg-transparent w-100 m-auto" onClick={(e) => e.stopPropagation()}>
        <div id="imageGallery">
          <GridGallery onSelect={handleImageSelect} />
        </div>
        <div style={{ marginTop: '20px'}}>
          <Button className='custom-button' onClick={handleRefreshClick}>Refresh</Button>
          <Button className='custom-button' onClick={onClose} style={{ marginInlineStart: '10px', marginInlineEnd: '10px'}}>Close</Button>
          <Button className='custom-button' onClick={handleUploadClick}>Upload your own</Button>
        </div>
      </div>
    </div>
  );
};

export default Modal;
