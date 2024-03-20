import React, { useState, useRef } from 'react';
import Modal from './Modal_ImageGallery';
import StegoMetrics from './StegoMetrics';
import SliderControl from './SliderControl';
import axios from 'axios'

import {Button} from 'react-bootstrap'

const serverURL = process.env.REACT_APP_NODE_SERVER_URI || '';

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
  const [sliderValue, setSliderValue] = useState(50);

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
  }

  const handleSliderChange = (value) => {
    setSliderValue(value)
  }

  const handleHideButtonClicked = async () => {
    setProcessing(true)

    if (coverImage && secretImage) {
      await axios.post(`${serverURL}/api/hide`, { coverImageData, secretImageData, sliderValue })
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
    } else {
      setProcessing(false)
    }
  }

  return (
    <div className='row'>
      <h2 className='col-12 custom-text-light mx-3 mt-1'>Image-in-Image Steganography</h2>
      <div className='row m-0 p-0 w-75 d-flex m-auto justify-content-around'>
        <div id="mainSection" className='row justify-content-around mx-auto align-middle'>
          <div id="secretImageSection" className='hoverShadow borderImage col-12 col-md-4 col-lg-2 p-0 my-3' onClick={() => handleItemClick("secretImage")}>
            {secretImage ? (
              <img
                ref={secretImageRef}
                src={secretImage.src}
                alt={secretImage.alt || ''}
                width={secretImage.width}
                height={secretImage.height}
              />
            ) : (
              <h3>Secret Image</h3>
            )}
          </div>
          <img className='col-12 col-sm-6 col-md-3 col-lg-1 p-0 my-auto' style={{maxHeight:'15vh'}} src='/images/plus_sign.svg' alt='Plus Sign'/>
          <div id="coverImageSection" className='hoverShadow borderImage col-12 col-md-4 col-lg-2 p-0 my-3' onClick={() => handleItemClick("coverImage")}>
            {coverImage ? (
              <img
                ref={coverImageRef}
                src={coverImage.src}
                alt={coverImage.alt || ''}
                width={coverImage.width}
                height={coverImage.height}
              />
            ) : (
              <h3>Cover Image</h3>
            )}
          </div>
          <img className='col-12 col-sm-6 col-md-3 col-lg-1 p-0 my-auto' style={{maxHeight:'15vh'}} src='/images/equals_sign.svg' alt='Equals Sign'/>
          <div id="stegoImageSection" className='borderImage hoverShadow col-8 col-sm-6 col-md-4 col-lg-2 p-0 my-3'>
            {stegoImage ? (
                <img
                  src={stegoImage}
                  alt={''}
                  width={coverImage.width}
                  height={coverImage.height}
                />
              ) : (
                <h3>Stego Image</h3>
              )}
          </div>
          <div className={`col-3 col-md-3 col-lg-2 p-0 ${stegoImage ? 'd-block' : 'd-none'}`}>
            {stegoImage ? (
              <StegoMetrics score={score} psnr={psnr} psnrScore={psnrScore} ssim={ssim} ssimScore={ssimScore}/>
            ) : (
              <></>
            )}
          </div>
          <div className='col-12 col-sm-4 col-md-3 col-lg-2 p-0 d-lg-none' height={200} width={200}></div>
        </div>
        <div className='row col-12 d-flex justify-content-around my-4'>
          <div className='col-12 col-md-4 w-75 d-md-flex justify-content-around'>
            <SliderControl onSliderChange={handleSliderChange} />
          </div>
          <Button className={`custom-button col-12 col-md-4 d-flex justify-content-center aligned-button`} onClick={handleHideButtonClicked}>
            {processing ? 'Processing...' : 'Hide!'}
          </Button>
        </div>
        <Modal isOpen={isModalOpen} 
              onClose={handleCloseModal} 
              selectedItem={selectedItem} 
              handleCoverImageSelect={handleCoverImageSelect} 
              handleSecretImageSelect={handleSecretImageSelect} />
    </div>
  </div>
  );
}

export default Hide;