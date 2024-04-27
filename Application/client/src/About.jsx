import React from 'react';
import SteGuzNetwork from './img/SteGuzNetwork.png';
import {Dropdown} from 'react-bootstrap'

const About = () => {
  return (
    <div id='about' className='border'>
      <div className='row about-wrapper col-10 w-100 px-3 m-0 pb-2 d-flex justify-content-around'>
        <Dropdown className="col-12 d-flex position-sticky justify-content-center d-sm-none">
          <Dropdown.Toggle className='w-100 custom-bg-fog mt-2 p-2' variant="light" id="dropdown-basic" style={{fontWeight: "bold"}}>
            Find Topic
          </Dropdown.Toggle>
          <Dropdown.Menu className='w-75'>
            <Dropdown.Item href='#steganography'>Steganography</Dropdown.Item>
            <Dropdown.Item href='#steguz'>SteGuz</Dropdown.Item>
            <Dropdown.Item href='#ssim'>SSIM</Dropdown.Item>
            <Dropdown.Item href='#psnr'>PSNR</Dropdown.Item>
          </Dropdown.Menu>
        </Dropdown>
        <div id='steganography' className='about-section col-12 col-md-5 mt-md-3 mt-0'>
            <h2>Steganography</h2>
            <h4>What is Steganography?</h4>
            <p>Steganography is the art of hiding information by embedding data within another medium in such a way that its mere presence goes undetected.
              By comparison, in cryptography, the information itself is not disguised, but changed through an algorithm to be unrecognizable. Watermarking
              is a related idea where data are embedded with some kind of identifier that signifies ownership or copyright. </p>

            <h4>Image-in-Image Steganography</h4>
            <p>
              Steganography can utilize various digital media, such as audio files, HTTP headers, plain text files, and even DNA.
              Image-in-image steganography involves hiding a secret image within a cover image.
              Ideally, the secret image undetectable to the naked eye, meaning only someone with the correct decoding mechanism can reveal the hidden information.</p>

            <h4>How does it work?</h4>
            <p>The process usually involves modifying the pixels of the cover image to encode the data of the secret image. This can be done using various methods,
              such as the Least Significant Bit (LSB) technique, where a variable number of the least significant bits of the cover image pixels are replaced with bits of the secret image.
              While LSB is effective, it is easy to detect with steganalysis tools, and its capacity is limited compared to a machine learning-oriented approach.
            </p>

            <h4>Applications of Image-in-Image Steganography</h4>
            <ul>
              <li>
                <h6><u>Covert Communication</u></h6>
                <p>The most famous applications of steganography involve covert communication, such as carving messages on the wooden backings of wax tablets in Ancient Rome, 
                  or embedding printed text in a space the size of a period (microdots), allowing for sensitive information to be discreetly communicated.
                </p>
              </li>
              <li>
                <h6><u>Protection of Intellectual Property</u></h6>
                <p>Steganography can be used to prove ownership or authenticity of materials, such as currency or driver's licenses.</p>
              </li>
              <li>
                <h6><u>Data Storage</u></h6>
                <p>Compared to cryptography, where we know there is some information there, just not what it is, with steganography's ability to hide information in an innocuous manner, 
                we can prevent prying eyes from even knowing there is information of interest present. For example, in medical imaging, a patient's information can be embedded in an image
                from a CT scan, exposing less useful information to potential bad actors.
                </p>
              </li>
            </ul>

            <h4>Significance in the Digital Age</h4>
            <p>Our modern world is vastly interconnected and more public and private data are being shared than ever before.
              Image-in-image steganography is becoming an essential tool, as it provides another layer of security for digital communications beyond encryption.
            </p>
          </div>
          <div id='steguz' className='about-section col-12 col-md-5 mt-md-3'>
            <h2>SteGuz Method</h2>
            <h4>Introduction</h4>
            <p>
              The SteGuz Method, developed by Dr. Amal Khalifa and Anthony Guzman at Purdue University Fort Wayne, represents a cutting-edge, machine learning-based approach to
              image-in-image steganography. This method employs three symmetry-adapted convolutional neural networks (CNNs) together to cover the entire steganographic process:
              preparing the image, hiding the image, and extracting the image.
            </p>
            
            <h4>Development and Innovations</h4>
            <p>
              A key breakthrough by Dr. Khalifa and her team, including graduate student Yashi Yadav, was the ability to use a secret image with the same dimensions as the cover image.
              By changing the CNNs to exclude noise layers and incorporating Least Significant Bit (LSB) techniques, the SteGuz Method enhances the efficacy and subtlety of the steganographic process.
            </p>
            
            <h4>The Process</h4>
            <p>
              The SteGuz Method operates through a three-stage process: First, the secret image is preprocessed to optimize it for embedding. Next, it is hidden within the cover image by the
              hiding network. Finally, the hidden image is extracted using the reveal network. This streamlined approach ensures both the cover and the secret images maintain high quality.
            </p>
            
            <img src={SteGuzNetwork} alt="Simplified view of the SteGuz Method" className="img-fluid my-3" />
            
            <p>
              The illustration provides a simplified overview of the SteGuz method's structure, showcasing the steps taken from image preparation to the revealing process.
              This website allows you to create your own Stego Images using the SteGuz method.
            </p>
            
            <h4>Model Training and Application</h4>
            <p>
              The deep learning model underpinning the SteGuz method was trained with a modest learning rate of 0.002, incorporating a crucial parameter, beta, to balance the output quality.
              With beta values adjustable between 0 and 1, users can influence the trade-off between the Stego Image's clarity and the extracted image's fidelity.
              The model's training involved around 15,000 images and completed in approximately one day, reflecting its efficiency and precision.
            </p>
            
            <p>
              On the "Hide Images" page, you're able to use various SteGuz models to create your own Stego Images.
            </p>
          </div>
          <div id='ssim' className='about-section col-12 col-md-5'>
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
          <div id='psnr' className='about-section col-12 col-md-5'>
            <h2>PSNR</h2>
            <h4>What is it?</h4>
            <p>PSNR (Peak Signal-Noise Ratio) is a commonly used metric to measure the quality of lossy compression of media such as videos and images.
              In the context of steganography, PSNR is utilized to evaluate the visual quality of the Stego Image (the image containing the hidden message) compared to the cover image.</p>
            <p>A higher PSNR value indicates better quality, meaning the difference between the original and Stego Image is less noticeable. A value of over 55 is considered indistinguishable from the original cover image.</p>

            <h4>How is PSNR calculated?</h4>
            <p>PSNR is calculated using the mean squared error (MSE) between the original and compressed or reconstructed images. The formula for PSNR is:</p>
            <p>PSNR = 20 * log<sub>10</sub>(MAX<sub>I</sub>) - 10 * log<sub>10</sub>(MSE)</p>
            <p>Where MAX<sub>I</sub> is the maximum possible pixel value of the image. The SteGuz method operates on 24-bit (RGB) images.
            You can think of each image as being composed of 3 layers (channels) - Red, Green, and Blue. On each layer, a pixel's maximum possible value is 255 (8 bits per pixel).
            </p>

            <h4>Why is PSNR important in steganography?</h4>
            <ul>
              <li>
                <h6><u>Quality Assessment</u></h6>
                <p>PSNR offers a simple way to measure the visual impact of the hidden data on the Stego Image. It helps in ensuring that the steganography process does not significantly degrade the quality of the original image.</p>
              </li>
              <li>
                <h6><u>Optimization</u></h6>
                <p>PSNR can inform the effectiveness of hiding images within another using various techniques, and in the context of the SteGuz method, tells us how our models and algorithms can be improved</p>
              </li>
            </ul>
          </div>
      </div>
    </div>
  )
}

export default About