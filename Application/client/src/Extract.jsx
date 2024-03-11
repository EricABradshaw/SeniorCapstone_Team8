import axios from 'axios';
import React, { useState } from 'react';
import { Button } from 'react-bootstrap'

const serverURL = process.env.REACT_APP_NODE_SERVER_URI;

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
  
        const response = await axios.post(`${serverURL}/api/extract`, formData);
  
        // Assuming the response contains the extracted secretImageData as base64
        const secretImageData = response.data.stegoImageData;
  
        // Create a Blob from the base64 string
        const blob = b64toBlob(secretImageData);
  
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
  
  // Helper function to convert base64 to Blob
  const b64toBlob = (b64Data, contentType = '', sliceSize = 512) => {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
  
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
  
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
  
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
  
    const blob = new Blob(byteArrays, { type: contentType });
    return blob;
  };
  

  return (
    <div className='row'>
      <h2 className='col-12 custom-text-light mx-3 mt-1'>Extraction</h2>
      <div className='row m-0 p-0 w-75 d-flex m-auto justify-content-around'>
        <div id="mainSection" className='row justify-content-around mx-auto align-middle'>
          <div className='hoverShadow borderImage col-12 col-md-4 col-lg-2 p-0 my-3' onClick={handleStegoImageClick}>
            {stegoImage ? (
              <img
                src={URL.createObjectURL(stegoImage)}
                alt="Stego"
                height={224}
                width={224}
              />
            ) : (
              <h3>Stego Image</h3>
            )}
          </div>

          <img className='col-12 col-sm-6 col-md-3 col-lg-1 p-0 my-auto' style={{maxHeight:'15vh'}} src='./images/arrow.svg' height={200} width={200} alt='Arrow'></img>
          <div className='hoverShadow borderImage col-12 col-md-4 col-lg-2 p-0 my-3'>
            {secretImage ? (
              <img
                src={secretImage}
                alt="Secret"
                height={224}
                width={224}
              />
            ) : (
              <h3>Secret Image</h3>
            )}
          </div>
        </div>
        <div className='row col-12 d-flex justify-content-around my-4'>
          <Button className={`custom-button col-12 col-md-4 d-flex justify-content-center aligned-button`} id='submitButton' onClick={handleExtractButtonClicked}>
            {processing ? 'Processing...' : 'Extract!'}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Extract;
