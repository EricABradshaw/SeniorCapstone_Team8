import React from 'react'

const SliderControl = ({ onSliderChange }) => {
  const handleSliderChange = (event) => {
    const value = parseInt(event.target.value)
    onSliderChange(value);
  }

  // fantastic > great > good > okay > acceptable
  // const printValue = (value) => {
  //   switch(value) {
  //     case 1:
  //       return <div><p>Acceptable StegoImage</p><p>Fantastic Extracted Image</p></div>
  //     case 2:
  //       return <div><p>Okay StegoImage</p><p>Great Extracted Image</p></div>
  //     case 3: 
  //       return <div><p>Good StegoImage</p><p>Good Extracted Image</p></div>
  //     case 4:
  //       return <div><p>Great StegoImage</p><p>Okay Extracted Image</p></div>
  //     case 5:
  //       return <div><p>Fantastic StegoImage</p><p>Acceptable StegoImage</p></div>
  //     default:
  //       return <div><p>Error</p></div>
  //   }
  // }

  return (
    <div id='sliderContents'>
      <h4>Better Extracted Image</h4>
      <div id='innerSlider'>
        <input
          type='range'
          id='betaSlider'
          min={1}
          max={100}
          onChange={handleSliderChange}
        />
      </div>
      <h4>Better StegoImage</h4>
    </div>
  )
}

export default SliderControl