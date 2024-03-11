import React from 'react'

const SliderControl = ({ onSliderChange }) => {
  const handleSliderChange = (event) => {
    const value = parseInt(event.target.value)
    onSliderChange(value);
  }

  return (
    <div className='row d-flex align-center w-100 my-5'>
      <h4 className='col-4 justify-content-end d-none d-md-flex custom-text-light'>Better Extracted Image</h4>
      <h4 className='col-12 d-flex d-md-none custom-text'>Beta Value</h4>
      <div className='col-12 col-md-4'>
        <input
          type='range'
          id='betaSlider'
          min={1}
          max={100}
          onChange={handleSliderChange}
          className='w-100'
        />
      </div>
      <h4 className='col-4 justify-content-start d-none d-md-flex custom-text-light'>Better StegoImage</h4>
    </div>
  )
}

export default SliderControl