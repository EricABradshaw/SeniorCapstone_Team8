import React, { useState} from 'react'
import './index.scss'

// import components
import ToggleButton from './ToggleButton';
import Hide from './Hide';
import Extract from './Extract';

function App() {
  const [isComponentAVisible, setComponentAVisible] = useState(true);

  //Callback function to decide which Elements are visible either hiding or extraction
  const handleToggle = () => {
    setComponentAVisible(!isComponentAVisible);
  };

  return (
    <div className="App">
      <span id="app-header">
        <h1>StegoSource.net</h1>
        <ToggleButton onToggle={handleToggle}/>
      </span>
      <div>
        {isComponentAVisible ? <Hide /> : <Extract />}
      </div>
    </div>
  )
}
export default App
