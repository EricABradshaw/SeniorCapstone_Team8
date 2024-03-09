import React, { useState} from 'react'
import './index.scss'
import './bsOverride.scss'
import { Button } from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'

// import components
import Hide from './Hide';
import Extract from './Extract';
import About from './About'
import HideText from './HideText'
import Recommend from './Recommend';

function App() {
  const [currentPage, setCurrentPage] = useState('main')
  const [activeButton, setActiveButton] = useState('main')

  const navigate = (page) => {
    setActiveButton(page)
    setCurrentPage(page)
  }

  let pageContent
  if (currentPage === 'about') {
    pageContent = <About />
  } else if (currentPage === 'text') {
    pageContent = <HideText />
  } else if (currentPage === 'recommend') {
    pageContent = <Recommend />
  } else if (currentPage === 'extract') {
    pageContent = <Extract />
  } else {
    pageContent = <Hide />
  }

  return (
    <div className="App bg-dark container-fluid" style={{height:'100vh'}}>
      <div className='mx-auto p-2 row w-100 border'>
        <h1 id='title' className='w-25 py-3 px-5'>StegoSource.net</h1>
        <div className='w-100 container-lg justify-content-around row' style={{margin:'auto'}}>
          <Button id='aboutBtn' className={`col-2 custom-button ${activeButton === "about" ? 'active' : ''}`} onClick={() => navigate('about')}>About</Button>
          <Button id='hideBtn' className={`col-2 custom-button ${activeButton === "main" ? 'active' : ''}`} onClick={() => navigate('main')}>Hide Images</Button>
          <Button id='textBtn' className={`col-2 custom-button ${activeButton === "text" ? 'active' : ''}`} onClick={() => navigate('text')}>Hide Text</Button>
          <Button id='extractBtn' className={`col-2 custom-button ${activeButton === "extract" ? 'active' : ''}`} onClick={() => navigate('extract')}>Extract</Button>
          <Button id='recommendBtn' className={`col-2 custom-button ${activeButton === "recommend" ? 'active' : ''}`} onClick={() => navigate('recommend')}>Recommend StegoImages</Button>
        </div>
      </div>
      <div>
        {pageContent}
      </div>
    </div>
  )
}
export default App
