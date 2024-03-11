import React, { useState, useRef } from 'react';
import Modal from './Modal_ImageGallery';
import StegoMetrics from './StegoMetrics';
import SliderControl from './SliderControl';
import axios from 'axios'
import {Button} from 'react-bootstrap'

const serverURL = process.env.REACT_APP_NODE_SERVER_URI;

const HideText = () => {
  const coverImageRef = useRef(null)
  const textRef = useRef(null)

  const [coverImage, setCoverImage] = useState(null);
  const [coverImageData, setCoverImageData] = useState(null)

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

    const text = textRef.current.value
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    canvas.width = 224
    canvas.height = 224

    ctx.fillStyle = 'gray'
    ctx.fillRect(0,0,canvas.width, canvas.height)

    ctx.fillStyle = 'black'
    ctx.font = '30px arial'

    wrapText(ctx, text, 10, 40, 220, 20)

    let textDataUrl = canvas.toDataURL('image/png')
    const paddingLength = textDataUrl.length % 4
    textDataUrl += '='.repeat(paddingLength)
    textDataUrl = textDataUrl.split(';base64,').pop()

    if (coverImage && textDataUrl) {
      console.log("Sending request...")
      await axios.post(`${serverURL}/api/hideText`, { coverImageData, textDataUrl, sliderValue })
        .then(response => {
          setStegoImage(`data:image/png;base64,${response.data.stegoImage.imageData}`)
          setPsnr(response.data.stegoImage.psnr)
          setSsim(response.data.stegoImage.ssim)
          handleRatings(response.data.stegoImage.ssim, response.data.stegoImage.psnr)
          setProcessing(false)
        })
        .catch(error => {
          console.error('Error sending data: ', error)
        })
    }
  }

  return (
    <div className='row'>
      <h2 className='col-12 custom-text-light mx-3 mt-1'>Text-in-Image Steganography</h2>
      <div className='row m-0 p-0 w-75 d-flex m-auto justify-content-around'>
        <div id="mainSection" className='row justify-content-around mx-auto align-middle'>
          <div className='col-12 col-md-4 col-lg-3 p-0 my-3'>
            <textarea ref={textRef} rows="4" style={{"resize":"vertical", "fontSize":20}}></textarea>
          </div>
          <img className='col-12 col-sm-6 col-md-3 col-lg-1 p-0 my-auto' style={{maxHeight:'15vh'}} src='/images/plus_sign.svg' height={200} width={200} alt='Plus Sign'></img>
          <div className='hoverShadow borderImage col-12 col-md-4 col-lg-2 p-0 my-3' onClick={() => handleItemClick("coverImage")}>
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
          <img className='col-12 col-sm-6 col-md-3 col-lg-1 p-0 my-auto' style={{maxHeight:'15vh'}} src='/images/equals_sign.svg' height={200} width={200} alt='Equals Sign'></img>
          <div className='hoverShadow borderImage col-12 col-md-4 col-lg-2 p-0 my-3'>
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
              handleCoverImageSelect={handleCoverImageSelect} />
      </div>
    </div> 
  );
}

const wrapText = (ctx, text, x, y, maxW, lineHeight) => {
  let words = text.split(' ')
  let line = ''

  for (let i = 0; i < words.length; i++) {
    let testLine = line + words[i] + ' '
    let metrics = ctx.measureText(testLine)
    let testWidth = metrics.width

    if (testWidth > maxW && i > 0) {
      ctx.fillText(line, x, y)
      line = words[i] + ' '
      y += lineHeight
    } else {
      line = testLine
    }
  }
  ctx.fillText(line, x, y)
}

export default HideText;