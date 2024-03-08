import axios from 'axios';
import React, { useState } from 'react';
import SliderControl from './SliderControl'; 

import oneStars from './img/1Stars.png';
import twoStars from './img/2Stars.png';
import threeStars from './img/3Stars.png';
import fourStars from './img/4Stars.png';
import fiveStars from './img/5Stars.png';


const serverURL = process.env.REACT_APP_NODE_SERVER_URI;
const NUMBER_OF_RECOMMENDATIONS = 20;

const Recommend = () => {
  const [secretImage, setSecretImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [imageUrls, setImageUrls] = useState([]);
  //
  const [sliderValue, setSliderValue] = useState(50);
  const [psnr, setPsnr] = useState(0);
  const [psnrScore, setPsnrScore] = useState(0);
  const [ssim, setSsim] = useState(0);
  const [ssimScore, setSsimScore] = useState(0);
  const [score, setScore] = useState(null);

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


///
const handleRatings = (ssim, psnr) => {    
  const score = (getPsnrScore(psnr) + getSsimScore(ssim)) / 2
  if (score < 1.5) {
    setScore("Awful")
  } else if (score < 2.5) {
    setScore("Bad")
  } else if (score < 3.5) {
    setScore("Fair")
  } else if (score < 4.5) {
    setScore("Good")
  } else {
    setScore("Excellent")
  }
}

const getPsnrScore = (psnr) => {
  if (psnr < 26) {
    setPsnrScore(1)
    return 1
  } else if (psnr < 27) {
    setPsnrScore(2)
    return 2
  } else if (psnr < 28) {
    setPsnrScore(3)
    return 3
  } else if (psnr < 29) {
    setPsnrScore(4)
    return 4
  } else {
    setPsnrScore(5)
    return 5
  }
}

const getSsimScore = (ssim) => {
  if (ssim < 0.94) {
    setSsimScore(1)
    return 1
  } else if (ssim < 0.95) {
    setSsimScore(2)
    return 2
  } else if (ssim < 0.97) {
    setSsimScore(3)
    return 3
  } else if (ssim < 0.98) {
    setSsimScore(4)
    return 4
  } else {
    setSsimScore(5)
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
    <div>
      <div id="mainSection" style={{ position: 'sticky', top: '110px', right: '0px', zIndex: 1 }}>
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
            <h1>Secret Image</h1>
          )}
        </div>
        <div className="filler"></div>
        <div className="filler"></div>
      </div>
      

      <div style={{ display: 'flex', marginTop: '32px', marginRight: '24px', marginLeft: '680px', position: 'fixed', zIndex: 1 }}>
      <SliderControl onSliderChange={(value) => setSliderValue(value)} />
      </div>
      <div id='submitButtonSection' style={{ position: 'fixed', top: '400px', zIndex: 1 }}>
        <button id='submitButton' onClick={handleRecommendButtonClicked}>
          {processing ? 'Processing...' : 'Recommend!'}
        </button>
      </div>
      
      <div style={{ marginTop: '220px', position: 'relative', zIndex: 2 }}>
        {imageUrls.map((imageUrl, index) => (
          <div style={{ position: 'relative', display: 'inline-block', margin: '10px' }} key={index}>
            <img src={imageUrl.src} alt={`Image ${index}`} height={224} width={224} />
            <img 
              src={starImages[imageUrl.starRatingImage]}  
              alt={`Rating stars`} 
              style={{ position: 'absolute', right: '1px', width: 'auto', height: 'auto', opacity: '0.8' }} 
            />
          </div>
        ))}
      </div>

    </div>
  );
  
};

export default Recommend;
