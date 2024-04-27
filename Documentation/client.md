# Deploying CNN-Driven Image Steganography Techniques in a Web Application

## Client Folder

**React Server**
The frontend of this application is developed using React. Development in React is done in the `Application/client/src` folder. 

The application is rendered in `./index.js`, which imports `App.js`. 

## App.js
App.js contains the primary routing of the frontend and the outlet of the various pages this application provides. Navigation utilizes React's useState to track the user's current page. As the user selects a different page, the state changes and the page is rendered.

### Styling
The following styling is imported in App.js, which is propagated to each JSX component.

- `index.scss`: This Sass file contains some basic styling that does not fall under the category of Bootstrap overriding.
- `bsOverride.scss`: The purpose of this Sass file is to override some of Bootstrap's default style. This file is primarily used to override the default colors Bootstrap uses. 

## Hide.jsx

### Overview
The `Hide` component is a React component responsible for implementing image-in-image steganography functionality. It allows users to hide a secret image within a cover image, producing a stego image. The component also provides options for adjusting parameters and evaluating the quality of the stego image.

### Imports and Setup
- React Hooks: `useState` and `useRef` for state management and DOM references.
- Components: `Modal`, `StegoMetrics`, and `SliderControl` for modal operations, displaying metrics, and slider functionality, respectively.
- Axios: HTTP client for sending requests to the server.
- React Bootstrap: `Button` for UI elements.

### State Management
- `useState` hooks manage the state of cover image, secret image, modal state, processing indication, slider value, stego image, PSNR, SSIM, and quality score.

### Refs
- `coverImageRef` and `secretImageRef` hold references to image DOM elements.

### Handlers and Functions
- `handleCoverImageSelect` and `handleSecretImageSelect` update state with selected images.
- `handleItemClick` opens modal with a specific item type.
- `handleCloseModal` closes the modal.
- `handleRatings`, `getPsnrScore`, and `getSsimScore` calculate and set quality scores.
- `handleSliderChange` updates slider value.
- `handleHideButtonClicked` sends cover and secret images to the server for processing, updates stego image and metrics.

### Component Layout
- Responsive layout using rows and columns.
- Display sections for secret image, cover image, stego image, and metrics.
- Button for initiating the hide process.
- Modal for selecting images.

### Component Structure
- **Main Section**: Contains display sections for images and metrics.
- **Slider Control**: Allows users to adjust parameters.
- **Hide Button**: Initiates the hiding process.
- **Modal**: Enables selecting images.

## Extract.jsx

### Overview

The `Extract` component is part of a React application that provides functionality to extract secret images from stego images. It allows users to select a stego image, processes it, and displays the extracted secret image.

### Features

- **Image Selection**: Users can click to select a stego image from their local device.
- **Extraction Processing**: After the stego image is selected, it is sent to a server where the extraction process is handled.
- **Display Results**: Displays the secret image extracted from the stego image.

### Usage

1. **Select a Stego Image**: Click on the designated area to select an image file from your device.
2. **Initiate Extraction**: Click the 'Extract!' button to start the extraction process.
3. **View the Secret Image**: Once processed, the secret image is displayed on the right side.

### Implementation Details

#### Environment Variables

- `REACT_APP_NODE_SERVER_URI`: The URI of the backend server where the extraction API is hosted.

#### Key Functions

##### `handleStegoImageClick`

Triggered by clicking the stego image section. Opens a file input to select an image and sets the selected image as the stego image.

##### `selectFile`

Creates a hidden file input for selecting images. Returns a `Promise` that resolves to the selected file.

##### `handleExtractButtonClicked`

Initiates the extraction process. It posts the stego image to the backend server and processes the response to display the secret ima


## About.jsx

The `About` component in this application serves as a comprehensive informational page regarding steganography and related technologies, including specific methods and measurement tools like SSIM and PSNR. It utilizes React and Bootstrap for styling and layout.

### Component Structure

This component is structured into several sections within a `<div>` container, each dedicated to different topics related to steganography. It includes a responsive dropdown menu for navigation on smaller screens.

#### Dependencies

- React
- react-bootstrap (for the Dropdown component)

#### Images

- `SteGuzNetwork.png`: An image illustrating the SteGuz Method.

### Sections

#### Dropdown Navigation

A responsive dropdown menu allows users to quickly navigate to different sections of the page:

- **Steganography**
- **SteGuz**
- **SSIM**
- **PSNR**

#### Steganography

This section introduces the concept of steganography, discusses its applications, and differentiates it from related concepts like cryptography. It covers:

- Definition and comparison with cryptography
- Image-in-image steganography methods
- Practical applications and significance in the digital age

#### SteGuz Method

Details the SteGuz Method developed by Dr. Amal Khalifa and Anthony Guzman, focusing on:

- Overview and innovations in the method
- Three-stage process: preparation, hiding, and extraction of images
- Model training and application details
- Illustration of the SteGuz Method (`SteGuzNetwork.png`)

#### SSIM (Structural Similarity Index Measure)

Explains SSIM, a measure used to assess the similarity between two images, particularly useful in steganography for:

- Definition and components (Luminance, Contrast, Structure)
- Application in evaluating steganography results

#### PSNR (Peak Signal-Noise Ratio)

Discusses PSNR, a common metric for measuring image quality after compression, focusing on:

- Definition and calculation
- Importance in steganography for quality assessment and optimization

### Usage

This component is intended to be used in educational or informational applications where users need detailed explanations about steganography and associated technologies. It serves both as an educational guide and a reference point for users exploring the application of these technologies in practical scenarios.

### Styling

Styling is primarily handled via Bootstrap classes for responsive layout and spaci

## HideText.jsx
### Overview

The `HideText` component is designed for text-in-image steganography. It allows users to embed text into an image (referred to as the cover image) while attempting to maintain the visual integrity of the original image. The component provides a user interface to select a cover image, input text, and view the modified image (stego image) alongside quality metrics such as PSNR and SSIM.

### Usage

To use the `HideText` component in your React application, follow these steps:

1. Import the component:

    ```javascript
    import HideText from './HideText';
    ```

2. Include the component in your JSX code:

    ```jsx
    <HideText />
    ```

### Features

- **Text Input**: Allows users to enter the text they wish to hide within the image.
- **Image Selection**: Users can select a cover image in which the text will be hidden.
- **Steganography Processing**: Combines the text and the image to produce a stego image.
- **Quality Metrics Display**: Shows the PSNR and SSIM scores to assess the quality of the stego image.
- **Responsive Slider Control**: Adjusts parameters affecting the embedding process, like robustness or visibility.

### Props

This component does not accept any props from its parent component.

### Dependencies

Ensure these dependencies are installed in your project:

- `axios`: For HTTP requests.
- `react-bootstrap`: For Bootstrap components in React.
- Custom components and hooks:
  - `Modal`: A modal component for image gallery.
  - `StegoMetrics`: Displays steganography metrics.
  - `SliderControl`: A custom slider for parameter adjustment.

### Component Structure

- **Text Area**: For inputting text to hide.
- **Cover Image Display**: Clickable area for selecting and displaying the cover image.
- **Stego Image Display**: Shows the resulting image after the text is embedded.
- **Slider Control**: For adjusting steganography parameters.
- **Hide Button**: Initiates the text hiding process.

### Server Interaction

The component communicates with a backend server via an API endpoint (`/api/hideText`). This endpoint expects the cover image data, text data, and slider value as inputs and returns the stego image data along with PSNR and SSIM values.

### Environment Variables

- `REACT_APP_NODE_SERVER_URI`: Specifies the server URL for the API calls.

### Error Handling

The component includes basic error handling for API requests, which logs errors to the console.

### Styling

Uses Bootstrap and custom CSS for styling. Adjust the CSS as needed to match your application's design.


## Recommend.jsx
### Overview
The `Recommend` component is a React component designed to provide image recommendations based on user input. It allows users to select a secret image, adjust a slider control, and receive recommended images.

### Usage
To use the `Recommend` component in your project, follow these steps:

1. Import the component:

    ```javascript
    import Recommend from './Recommend';
    ```

2. Include the `Recommend` component in your JSX code:

    ```jsx
    <Recommend />
    ```

3. Customize the component as needed by modifying its props and styling.

### Props

The `Recommend` component does not accept any props.

### Dependencies

The `Recommend` component relies on the following dependencies:

- axios: For making HTTP requests to the server.
- react-bootstrap: For styling and UI components.
- SliderControl component: A custom slider control component (not provided in this documentation).

Ensure that these dependencies are installed in your project before using the `Recommend` component.

### Functionality

#### Selecting a Secret Image

- Users can click on the "Secret Image" area to select an image file from their device.
- The selected image will be displayed in the "Secret Image" area.

#### Adjusting the Slider Control

- Users can adjust the slider control to set a value between 0 and 100.
- This value represents a parameter used for image recommendation (e.g., image quality, similarity threshold).

#### Processing Recommendations

- When users click the "Recommend!" button, the component sends a request to the server with the selected secret image and slider value.
- The server processes the request and returns a set of recommended images.
- The recommended images, along with their ratings, are displayed below the "Secret Image" area.

### File Structure

- `Recommend.jsx`: The main component file containing the `Recommend` component definition.
- `SliderControl.jsx`: A custom slider control component (not provided in this documentation).
- `img/`: Directory containing star rating images used in the component.

### Environment Variables

- `REACT_APP_NODE_SERVER_URI`: Environment variable specifying the URL of the server. If not provided, the component falls back to a default value.

Ensure that the server specified by the `REACT_APP_NODE_SERVER_URI` environment variable is running and configured to handle image recommendation requests.

## Modal_ImageGallery.jsx (Modal)
### Overview
The `Modal` component is a React component used to display a modal overlay with options for selecting images from a grid gallery, refreshing the gallery, closing the modal, and uploading custom images.

### Imports and Setup
- React Hooks: `useState` for managing state.
- Components: `GridGallery` for displaying image grid and `Button` from React Bootstrap for UI elements.

### State Management
- `useState` hook manages the state of the refresh key, which triggers re-rendering of the grid gallery.

### Handlers and Functions
- `handleRefreshClick` increments the refresh key state, forcing the grid gallery to refresh.
- `handleUploadClick` allows users to upload custom images by selecting files and converting them to base64 format.
- `selectFile` opens a file dialog for selecting files.
- `fileToBase64` converts the selected file to base64 format with specified dimensions.
- `handleImageSelect` handles the selection of images from the grid gallery and passes them to the parent component based on the selected item type.

### Component Layout
- Modal overlay with transparent background.
- Modal content with grid gallery for image selection and buttons for actions.
- Buttons for refreshing the gallery, closing the modal, and uploading custom images.

### Component Structure
- **Modal Overlay**: Wraps the modal content and closes the modal when clicked.
- **Modal Content**: Contains the grid gallery and action buttons.
- **Grid Gallery**: Displays images for selection.
- **Refresh Button**: Triggers gallery refresh.
- **Close Button**: Closes the modal.
- **Upload Button**: Allows users to upload custom images.