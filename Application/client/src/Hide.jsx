import React, { useState } from 'react';
import Modal from './Modal_ImageGallery';

const Hide = () => {
  const [coverImage, setCoverImage] = useState(null);
  const [secretImage, setSecretImage] = useState(null);
  const [isModalOpen, setModalOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);

  // Callback function to be passed to GridGallery
  const handleCoverImageSelect = (image) => {
    console.log('Selected Image:', image);
    setCoverImage(image);
  };

  const handleSecretImageSelect = (image) =>{
    console.log('Secret Image:', image);
    setSecretImage(image);
  }

  const handleItemClick = (item) => {
    setSelectedItem(item);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
    setSelectedItem(null);
  };

  return (
    <div>
      <div id="mainSection">
        <div className="filler"></div>
        <div id="secretImageSection" onClick={() => handleItemClick("secretImage")}>
        {secretImage ? (
            <img
              src={secretImage.src}
              alt={secretImage.alt || ''}
              width={secretImage.width}
              height={secretImage.height}
            />
          ) : (
            <h1>Secret Image</h1>
          )}
        </div>
        <img src='./images/plus_sign.svg' height={200} width={200} alt='Plus Sign'></img>
        <div id="coverImageSection" onClick={() => handleItemClick("coverImage")}>
          {coverImage ? (
            <img
              src={coverImage.src}
              alt={coverImage.alt || ''}
              width={coverImage.width}
              height={coverImage.height}
            />
          ) : (
            <h1>Cover Image</h1>
          )}
        </div>
        <img src='./images/equals_sign.svg' height={200} width={200} alt='Equals Sign'></img>
        <div id="stegoImageSection">
          <h1>Stego Image?</h1>
        </div>
        <div className="filler"></div>
      </div>
      <div id='submitButtonSection'>
        <button id='submitButton'>Hide!</button>
      </div>
      <Modal isOpen={isModalOpen} onClose={handleCloseModal} selectedItem={selectedItem} handleCoverImageSelect={handleCoverImageSelect} handleSecretImageSelect={handleSecretImageSelect}/>
    </div>
  );
}

export default Hide;