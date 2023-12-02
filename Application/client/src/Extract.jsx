import axios from 'axios';
import React, { useState } from 'react';

const Extract = () => {
  const [stegoImage, setStegoImage] = useState(null);
  const [secretImage, setSecretImage] = useState(null);
  const [processing, setProcessing] = useState(false);

  const handleStegoImageClick = async () => {
    try {
      const selectedFile = await selectFile();
      if (selectedFile) {
        setStegoImage(selectedFile);
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

  const handleExtractButtonClicked = async () => {
    setProcessing(true);
  
    if (stegoImage) {
      try {
        const formData = new FormData();
        formData.append('stegoImage', stegoImage);
  
        const response = await axios.post('http://localhost:9000/api/extract', formData);
  
        // Assuming the response contains the extracted secretImage data
        const secretImageData = response.data.stegoImageData;
  
        // Convert base64 to binary data
        const binaryData = atob(secretImageData);
  
        // Create an array buffer from binary data
        const arrayBuffer = new ArrayBuffer(binaryData.length);
        const uint8Array = new Uint8Array(arrayBuffer);
  
        for (let i = 0; i < binaryData.length; i++) {
          uint8Array[i] = binaryData.charCodeAt(i);
        }
  
        // Create a Blob from the array buffer
        const blob = new Blob([arrayBuffer]);
  
        // Create a data URL for the blob
        const dataUrl = URL.createObjectURL(blob);
  
        // Set the secretImage state with the data URL
        setSecretImage(dataUrl);
  
        setProcessing(false);
      } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error);
        setProcessing(false);
      }
    }
  };
  

  return (
    <div>
      <div id="mainSection">
        <div className="filler"></div>
        <div className="filler"></div>
        <div id="stegoImageSection" onClick={handleStegoImageClick}>
          {stegoImage ? (
            <img
              src={URL.createObjectURL(stegoImage)}
              alt="Stego Image"
              height={224}
              width={224}
            />
          ) : (
            <h1>Stego-Image</h1>
          )}
        </div>

        <img src='./images/arrow.svg' height={200} width={200} alt='Arrow'></img>
        <div id="secretImageSection">
          {secretImage ? (
            <img
              src={secretImage}
              alt="Secret Image"
              height={224}
              width={224}
            />
          ) : (
            <h1>Secret Image</h1>
          )}
        </div>
        <div className="filler"></div>
        <div className="filler"></div>
      </div>
      <div id='submitButtonSection'>
        <button id='submitButton' onClick={handleExtractButtonClicked}>
          {processing ? 'Processing...' : 'Extract!'}
        </button>
      </div>
    </div>
  );
};

export default Extract;
