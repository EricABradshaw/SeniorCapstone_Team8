import React from 'react'

const SliderControl = ({ onSliderChange }) => {
  const handleSliderChange = (event) => {
    const value = parseInt(event.target.value)
    onSliderChange(value);
  }

  return (
    <div id='sliderContents'>
      <h4 className='sliderText'>Better Extracted Image</h4>
      <div id='innerSlider'>
        <input
          type='range'
          id='betaSlider'
          min={1}
          max={100}
          onChange={handleSliderChange}
        />
      </div>
      <h4 className='sliderText'>Better StegoImage</h4>
    </div>
  )
}

export default SliderControl