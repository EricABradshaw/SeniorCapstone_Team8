import axios from 'axios';
import React, { useState } from 'react';
import {Button} from 'react-bootstrap';
import SliderControl from './SliderControl'; 

import oneStars from './img/1Stars.png';
import twoStars from './img/2Stars.png';
import threeStars from './img/3Stars.png';
import fourStars from './img/4Stars.png';
import fiveStars from './img/5Stars.png';

const serverURL = process.env.REACT_APP_NODE_SERVER_URI || '';
const NUMBER_OF_RECOMMENDATIONS = 20;

const Recommend = () => {
  const [secretImage, setSecretImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [imageUrls, setImageUrls] = useState([]);
  const [sliderValue, setSliderValue] = useState(50);

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

const handleSliderChange = (value) => {
  setSliderValue(value);
};

const getPsnrScore = (psnr) => {
  if (psnr < 26) {
    return 1
  } else if (psnr < 27) {
    return 2
  } else if (psnr < 28) {
    return 3
  } else if (psnr < 29) {
    return 4
  } else {
    return 5
  }
}

const getSsimScore = (ssim) => {
  if (ssim < 0.94) {
    return 1
  } else if (ssim < 0.95) {
    return 2
  } else if (ssim < 0.97) {
    return 3
  } else if (ssim < 0.98) {
    return 4
  } else {
    return 5
  }
}

const starImages = {
  '1Stars.png': oneStars,
  '2Stars.png': twoStars,
  '3Stars.png': threeStars,
  '4Stars.png': fourStars,
  '5Stars.png': fiveStars,
};

///

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
        const requestURLs = Array.from({ length: NUMBER_OF_RECOMMENDATIONS }, () => `${serverURL}/api/recommendation`);
        // Make concurrent POST requests using Axios
        //const sliderValue = 75;
        const responses = await Promise.all(requestURLs.map(url => axios.post(url, {secretImage, sliderValue})));
        // Extract data from each response and do something with it

        const modifiedImageUrls = responses.map(response => {
          // Calculate ratings
          const psnrRating = getPsnrScore(response.data.stegoImage.psnr);
          const ssimRating = getSsimScore(response.data.stegoImage.ssim);
          const averageScore = Math.floor((psnrRating + ssimRating) / 2);

          // Determine star image to use based on average score
          const starRatingImage = `${averageScore}Stars.png`;
  
          return {
            src: `data:image/png;base64,${response.data.stegoImage.imageData}`,
            starRatingImage: starRatingImage
          }
        });

        setImageUrls(modifiedImageUrls)
        setProcessing(false);
      } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error);
        setProcessing(false);
      }
    }
  };

  return (
    <div className='row w-100 d-flex m-auto'>
      <div className={`${processing ? 'd-none' : 'd-block'}`}>
        {imageUrls.map((imageUrl, index) => (
          <div className='position-relative d-inline-block m-1' key={index}>
            <img src={imageUrl.src} alt={`Im${index}`} />
            <img 
              src={starImages[imageUrl.starRatingImage]}  
              alt={`Rating stars`} 
              className={'hover-hide'}
            />
          </div>
        ))}
      </div>
      <div id="mainSection" className='col'>
        <div className='borderImage hoverShadow' onClick={handleSecretImageClick}>
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
          <div className='col-12 col-md-4 w-75 d-md-flex justify-content-around'>
            <SliderControl onSliderChange={handleSliderChange} />
          </div>
          <Button className={`custom-button col-12 col-md-4 d-flex justify-content-center aligned-button`} onClick={handleRecommendButtonClicked}>
            {processing ? 'Processing...' : 'Recommend!'}
          </Button>
        </div>
    </div>
  );
  
};

export default Recommend;
