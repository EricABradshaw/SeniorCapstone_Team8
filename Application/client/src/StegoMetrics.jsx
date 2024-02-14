import React, { useState } from 'react'

const StegoImage = (props) => {
  const [isVisible, setIsVisible] = useState(false);

  const toggleVisibility = () => {
    console.log(isVisible)
    setIsVisible(!isVisible)
  }

  return (
    <div className="stegoMetrics">
      <div className={`content ${isVisible ? 'visible' : 'hidden'}`}>
        <p>PSNR: {props.psnr}</p>
        <p>Score: {props.psnrScore}</p>
        &nbsp;
        <p>SSIM: {props.ssim}</p>
        <p>Score: {props.ssimScore}</p>
        <h2>{props.score}</h2>
      </div>
      <div id='visButtonContainer'>
          <div className='visibilityTrigger'>
            <button onClick={toggleVisibility}>
              ?
            </button>
          </div>
        </div>
    </div>
  )
}


export default StegoImage