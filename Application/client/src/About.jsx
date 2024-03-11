import React from 'react';

const About = () => {
  return (
    <div id='about' className='mt-4'>
      <div className='row about-wrapper col-10 w-100 p-3 m-0 d-flex justify-content-around'>
        <div className='about-section col-12 col-md-5'>
          <h2>Steganography</h2>
        </div>
        <div className='about-section col-12 col-md-5'>
          <h2>SteGuz Method</h2>
        </div>
        <div className='about-section col-12 col-md-5'>
          <h2>SSIM</h2>
          <h4>What is it?</h4>
          <p>SSIM &#40;Structural Similarity Index Measure&#41; is used to measure the similarity between two images.</p>
          <p>In steganography, it can be used to measure how closely the extracted image resembles the original secret image!</p>
          <p>The closer the SSIM value is to 1, the closer the extracted image will look to the hidden secret image. When SSIM = 1, the images are exactly the same</p>
          <h4>What is the SSIM composed of?</h4>
          <ul>
            <li>
              <h6><u>Luminance</u></h6>
              <p>This is how bright or dark an image appears overall.</p>
              <p>For use in the SSIM, luminance is measured by the average intensity of all pixels in the image.</p>
            </li>
            <li>
              <h6><u>Contrast</u></h6>
              <p>When dealing with image processing, contrast is the color differentiation between different parts of the image.</p>
              <p>The constrast value for use in the SSIM takes the standard deviation of pixel values.</p>
            </li>
            <li>
              <h6><u>Structure</u></h6>
              <p>Parts of the image can be broken down into partitions based on features such as color, texture, or intensity.</p>
              <p>Structure is measured by comparing similarities of local patterns between two images.</p>
            </li>
          </ul>
        </div>
        <div className='about-section col-12 col-md-5'>
          <h2>PSNR</h2>
          <h4>What is it?</h4>
        </div>
      </div>
    </div>
  )
}

export default About