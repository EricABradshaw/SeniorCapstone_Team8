import React, { useState } from 'react';

const ToggleButton = ({ onToggle }) => {
  const [isToggled, setToggled] = useState(false);

  const handleToggle = () => {
    setToggled(!isToggled);
    onToggle();
  };

  return (
    <div className="toggle-container">
      <button onClick={handleToggle} className={isToggled ? 'active' : ''}>
        {isToggled ? 'Hide An Image Instead' : 'Extract An Image Instead'}
      </button>
    </div>
  );
};

export default ToggleButton;
