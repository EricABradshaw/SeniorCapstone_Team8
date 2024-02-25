import React, { useState} from 'react'
import './index.scss'

// import components
import ToggleButton from './ToggleButton';
import Hide from './Hide';
import Extract from './Extract';
import About from './About'
import HideText from './HideText'

function App() {
  const [isComponentAVisible, setComponentAVisible] = useState(true);
  const [currentPage, setCurrentPage] = useState('main')

  const navigate = (page) => {
    setCurrentPage(page)
  }

  let pageContent
  if (currentPage === 'about') {
    pageContent = <About />
  } else if (currentPage === 'main') {
    pageContent = isComponentAVisible ? <Hide /> : <Extract />
  } else if (currentPage === 'text') {
    pageContent = <HideText />
  }

  //Callback function to decide which Elements are visible either hiding or extraction
  const handleToggle = () => {
    setComponentAVisible(!isComponentAVisible);
  };

  return (
    <div className="App">
      <span id="app-header">
        <h1>StegoSource.net</h1>
        <button onClick={() => navigate('about')}>About</button>
        {currentPage === 'main' ? (
          <button onClick={() => navigate('text')}>Hide Text</button>
        ) : (
          <button onClick={() => navigate('main')}>Hide Image</button>
        )}
        <ToggleButton onToggle={handleToggle}/>
      </span>
      <div>
        {pageContent}
      </div>
    </div>
  )
}
export default App
