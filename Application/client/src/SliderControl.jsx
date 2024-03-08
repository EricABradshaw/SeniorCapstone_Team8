import React from 'react'

const SliderControl = ({ onSliderChange }) => {
  const handleSliderChange = (event) => {
    const value = parseInt(event.target.value)
    onSliderChange(value);
  }

  return (
    <div id='sliderContents' style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <h4 className='sliderText' style={{ marginRight: '20px' }}>Better Extracted Image</h4>
      <div id='innerSlider' style={{ flex: 1 }}>
        <input
          type='range'
          id='betaSlider'
          min={1}
          max={100}
          onChange={handleSliderChange}
          style={{ width: '100%' }} // Ensure the slider takes the full width of its container
        />
      </div>
      <h4 className='sliderText' style={{ marginLeft: '20px' }}>Better StegoImage</h4>
    </div>
  )
}

export default SliderControl