import React, { useState} from 'react'
import { Button, Dropdown } from 'react-bootstrap'

import './index.scss'
import './bsOverride.scss'
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
    <div className="App bg-dark container-fluid p-0" style={{height:'100%', minHeight:'100vh', overflow:'hidden'}}>
      <div className='mx-auto p-2 row w-100 bg-black border-bottom'> 
        <h1 id='title' className='w-25 py-3 px-5'>StegoSource.net</h1>
        <Dropdown className="col-12 d-block d-sm-none">
          <Dropdown.Toggle variant="dark" id="dropdown-basic">
            Menu
          </Dropdown.Toggle>
          <Dropdown.Menu>
            <Dropdown.Item onClick={() => navigate('about')}>About</Dropdown.Item>
            <Dropdown.Item onClick={() => navigate('main')}>Hide Images</Dropdown.Item>
            <Dropdown.Item onClick={() => navigate('text')}>Hide Text</Dropdown.Item>
            <Dropdown.Item onClick={() => navigate('extract')}>Extract</Dropdown.Item>
            <Dropdown.Item onClick={() => navigate('recommend')}>Recommend StegoImages</Dropdown.Item>
          </Dropdown.Menu>
        </Dropdown>
        <div className='d-none d-sm-flex w-100 justify-content-around row m-auto'>
          <Button id='aboutBtn' className={`col-sm-6 col-md-3 col-lg-2 custom-button ${activeButton === "about" ? 'active' : ''}`} onClick={() => navigate('about')}>About</Button>
          <Button id='hideBtn' className={`col-sm-6 col-md-3 col-lg-2 custom-button ${activeButton === "main" ? 'active' : ''}`} onClick={() => navigate('main')}>Hide Images</Button>
          <Button id='textBtn' className={`col-sm-6 col-md-3 col-lg-2 custom-button ${activeButton === "text" ? 'active' : ''}`} onClick={() => navigate('text')}>Hide Text</Button>
          <Button id='extractBtn' className={`col-sm-6 col-md-3 col-lg-2 custom-button ${activeButton === "extract" ? 'active' : ''}`} onClick={() => navigate('extract')}>Extract</Button>
          <Button id='recommendBtn' className={`col-sm-6 col-md-6 col-lg-2 custom-button ${activeButton === "recommend" ? 'active' : ''}`} onClick={() => navigate('recommend')}>Recommend</Button>
        </div>
      </div>
      <div>
        {pageContent}
      </div>
    </div>
  )
}
export default App
