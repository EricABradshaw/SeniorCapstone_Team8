import React, { useState, useRef } from 'react';
import Modal from './Modal_ImageGallery';
import axios from 'axios'

const Hide = () => {
  const coverImageRef = useRef(null)
  const secretImageRef = useRef(null)
  const [coverImage, setCoverImage] = useState(null);
  const [coverImageData, setCoverImageData] = useState(null)
  const [secretImage, setSecretImage] = useState(null);
  const [secretImageData, setSecretImageData] = useState(null)
  const [isModalOpen, setModalOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [processing, setProcessing] = useState(false);

  const [stegoImage, setStegoImage] = useState(null);

  // Callback function to be passed to GridGallery
  const handleCoverImageSelect = (image) => {
    console.log('Selected Image:', image);
    setCoverImage(image);
    setCoverImageData(image.src)
  };

  const handleSecretImageSelect = (image) => {
    console.log('Secret Image:', image);
    setSecretImage(image);
    setSecretImageData(image.src)
  }

  const handleItemClick = (item) => {
    setSelectedItem(item);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
    setSelectedItem(null);
  };

  const handleHideButtonClicked = async () => {
    setProcessing(true)

    if (coverImage && secretImage) {
      console.log("Sending request...")
      await axios.post('http://localhost:9000/api/hide', { coverImageData, secretImageData })
        .then(response => {
          console.log(response)
          setStegoImage(`data:image/png;base64,${response.data.stegoImage}`)
          setProcessing(false)
        })
        .catch(error => {
          console.error('Error sending data: ', error)
        })
    }
  }

  return (
    <div>
      <div id="mainSection">
        <div className="filler"></div>
        <div id="secretImageSection" onClick={() => handleItemClick("secretImage")}>
          {secretImage ? (
            <img
              ref={secretImageRef}
              src={secretImage.src}
              alt={secretImage.alt || ''}
              width={secretImage.width}
              height={secretImage.height}
            />
          ) : (
            <h1>Secret Image</h1>
          )}
        </div>
        <img src='/images/plus_sign.svg' height={200} width={200} alt='Plus Sign'></img>
        <div id="coverImageSection" onClick={() => handleItemClick("coverImage")}>
          {coverImage ? (
            <img
              ref={coverImageRef}
              src={coverImage.src}
              alt={coverImage.alt || ''}
              width={coverImage.width}
              height={coverImage.height}
            />
          ) : (
            <h1>Cover Image</h1>
          )}
        </div>
        <img src='/images/equals_sign.svg' height={200} width={200} alt='Equals Sign'></img>
        <div id="stegoImageSection">
          {stegoImage ? (
              <img
                src={stegoImage}
                alt={''}
                width={coverImage.width}
                height={coverImage.height}
              />
            ) : (
              <h1>Stego Image</h1>
            )}
        </div>
        <div className="filler"></div>
      </div>
      <div id='submitButtonSection'>
        <button id='submitButton' onClick={handleHideButtonClicked}>
          {processing ? 'Processing...' : 'Hide!'}
        </button>
      </div>
      <Modal isOpen={isModalOpen} onClose={handleCloseModal} selectedItem={selectedItem} handleCoverImageSelect={handleCoverImageSelect} handleSecretImageSelect={handleSecretImageSelect} />
    </div>
  );
}

export default Hide;