import React from 'react';

const Extract = () => {

  function upload() {
    let input = document.createElement('input');
    input.type = 'file';

    input.onchange = (event) => {
      const stego = event.target.files[0];

      if (stego) {
        console.log('Selected file:', stego);

        const newImage = document.createElement('img');
        newImage.src = URL.createObjectURL(stego); 
        newImage.height = 224;
        newImage.width = 224;

        const parent = document.getElementById('stegoImageSection');
        parent.innerHTML = ''; 
        parent.appendChild(newImage);
      }
    };

    input.click();
  }

  return (
    <div>
      <div id="mainSection">
        <div className="filler"></div>
        <div className="filler"></div>
        <div id="stegoImageSection" onClick={upload}>
          <h1>Stego-Image</h1>
        </div>

        <img src='./images/arrow.svg' height={200} width={200} alt='Arrow'></img>
        <div id="secretImageSection">
          <h1>Secret Image</h1>
        </div>
        <div className="filler"></div>
        <div className="filler"></div>
      </div>
      <div id='submitButtonSection'>
        <button id='submitButton'>Extract!</button>
      </div>
    </div>
  );
}

export default Extract;