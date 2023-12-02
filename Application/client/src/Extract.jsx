import React from 'react';

const Extract = () => {

  return (
    <div>
      <div id="mainSection">
        <div className="filler"></div>
        <div className="filler"></div>
        <div id="stegoImageSection">
          <h1>Stego Image</h1>
        </div>
        <img src='/images/arrow.svg' height={200} width={200} alt='Arrow Sign'></img>
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