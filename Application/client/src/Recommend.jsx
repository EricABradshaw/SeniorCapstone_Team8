import axios from 'axios';
import React, { useState } from 'react';

const serverURL = process.env.REACT_APP_NODE_SERVER_URI;

const Recommend = () => {
  const [secretImage, setSecretImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [imageUrls, setImageUrls] = useState([]);

  const handleSecretImageClick = async () => {
    try {
      const selectedFile = await selectFile();
      if (selectedFile) {
        let base64String = await fileToBase64(selectedFile, 224, 224);
        setSecretImage(base64String);
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

  const handleRecommendButtonClicked = async () => {
    setProcessing(true);
  
    if (secretImage) {
      try {
        // Create an array with the same request body repeated n times
        const requestURLs = Array.from({ length: 50 }, () => `${serverURL}/api/recommendation`);
        // Make concurrent POST requests using Axios
        const sliderValue = 75;
        const responses = await Promise.all(requestURLs.map(url => axios.post(url, {secretImage, sliderValue})));
        // Extract data from each response and do something with it
        const imageUrls = responses.map(response => {
          return {
            src: `data:image/png;base64,${response.data.stegoImage.imageData}`, 
          }
        });
        
        setImageUrls(imageUrls)
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
        <div className='borderImage hoverShadow' onClick={handleSecretImageClick}>
          {secretImage ? (
            <img
              src={secretImage}
              alt="Secret Image"
              height={224}
              width={224}
            />
          ) : (
            <h1>Secret-Image</h1>
          )}
        </div>

        <div className="filler"></div>
        <div className="filler"></div>
      </div>
      <div id='submitButtonSection'>
        <button id='submitButton' onClick={handleRecommendButtonClicked}>
          {processing ? 'Processing...' : 'Recommend!'}
        </button>
      </div>
      <div>
        {imageUrls.map((imageUrl, index) => (
          <img src={imageUrl.src} alt={`Image ${index}`} height={224} width={224} />
        ))}
      </div>
    </div>
  );
};

export default Recommend;
