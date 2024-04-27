import React, { useState } from 'react'
import { Button } from 'react-bootstrap'

const StegoImage = (props) => {
  const [isVisible, setIsVisible] = useState(false);

  const toggleVisibility = () => {
    setIsVisible(!isVisible)
  }

  return (
    <div className='row d-flex m-0 p-0'>
      <div className='position-relative col-2' style={{zIndex:'1'}}>
        <Button className='custom-button' onClick={toggleVisibility}>
          ?
        </Button>
      </div>
      <div className='col-2'></div>
      <div className='position-relative col-8'>
        <div className={`position-absolute custom-text border rounded p-2 ${isVisible ? 'd-inline-block position-static' : 'd-none'}`}>
          <p>PSNR: {props.psnr}</p>
          <p>Score: {props.psnrScore}</p>
          <p>SSIM: {props.ssim}</p>
          <p>Score: {props.ssimScore}</p>
          <h2>{props.score}</h2>
        </div>
      </div>
    </div>
  )
}


export default StegoImage