import React, { useState, useRef } from 'react';
import Modal from './Modal_ImageGallery';
import StegoMetrics from './StegoMetrics';
import SliderControl from './SliderControl';
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
  const [sliderValue, setSliderValue] = useState(3);

  const [stegoImage, setStegoImage] = useState(null);
  const [psnr, setPsnr] = useState(0);
  const [psnrScore, setPsnrScore] = useState(0);
  const [ssim, setSsim] = useState(0);
  const [ssimScore, setSsimScore] = useState(0);
  const [score, setScore] = useState(null);

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

  const handleSliderChange = (value) => {
    setSliderValue(value)
  }

  const handleHideButtonClicked = async () => {
    setProcessing(true)

    if (coverImage && secretImage) {
      console.log("Sending request...")
      await axios.post('http://localhost:9000/api/hide', { coverImageData, secretImageData, sliderValue })
        .then(response => {
          console.log(response)
          setStegoImage(`data:image/png;base64,${response.data.stegoImage.imageData}`)
          setPsnr(response.data.stegoImage.psnr)
          setSsim(response.data.stegoImage.ssim)
          handleRatings(response.data.stegoImage.ssim, response.data.stegoImage.psnr)
          console.log(`${response.data.stegoImage.ssim} ${response.data.stegoImage.psnr}`)
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
        <div id="secretImageSection" className='hoverShadow borderImage' onClick={() => handleItemClick("secretImage")}>
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
        <div id="coverImageSection" className='hoverShadow borderImage' onClick={() => handleItemClick("coverImage")}>
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
        <div id="stegoImageContainer">
          {stegoImage ? (
            <div>
              <StegoMetrics score={score} psnr={psnr} psnrScore={psnrScore} ssim={ssim} ssimScore={ssimScore}/>
            </div>
          ) : (
            <></>
          )}
          <div id="stegoImageSection" className='borderImage'>
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
        </div>
        <div className="filler"></div>
      </div>
      <div id='sliderContainer'>
        <SliderControl onSliderChange={handleSliderChange} />
      </div>
      <div id='submitButtonSection'>
        <button id='submitButton' onClick={handleHideButtonClicked}>
          {processing ? 'Processing...' : 'Hide!'}
        </button>
      </div>
      <Modal isOpen={isModalOpen} 
             onClose={handleCloseModal} 
             selectedItem={selectedItem} 
             handleCoverImageSelect={handleCoverImageSelect} 
             handleSecretImageSelect={handleSecretImageSelect} />
    </div>
  );
}

export default Hide;