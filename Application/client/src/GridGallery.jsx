import React from 'react';
import { Gallery } from "react-grid-gallery";

const NUM_IMAGES_REQUEST_FROM_API = 16;

function getImages(numberOfImages) {
  const apiUrl = 'https://picsum.photos/224/224';
  return new Promise((resolve, reject) => {
    if (!Number.isInteger(numberOfImages) || numberOfImages <= 0) {
      return reject(new Error('Invalid number of images'));
    }
    // Make a request to the image API
    const imagePromises = [];

    for (let i = 0; i < numberOfImages; i++) {
      imagePromises.push(fetch(`${apiUrl}?random=${i}`).then(response => response.blob()));
    }

    Promise.all(imagePromises)
      .then(blobs => {
        resolve(blobs);
      })
      .catch(error => {
        reject(error);
      });
  });
}

class GridGallery extends React.Component {
  constructor() {
    super();
    this.state = {
      images: []
    };
  }

  componentDidMount() {
    console.log('Component mounted')
    this.loadImages()
  }

  loadImages = () => {
    getImages(NUM_IMAGES_REQUEST_FROM_API)
    .then(blobs => Promise.all(blobs.map(blob => this.blobToBase64(blob))))
    .then(base64Images => {
      const imageArray = base64Images.map(base64data => ({
        src: base64data,
        width: 224,
        height: 224,
      }));
      this.setState({ images: imageArray });
    })
    .catch(error => {
      console.error(error.message);
    });
  }

  blobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = function() {
        const base64data = reader.result;
        resolve(base64data);
      }
      reader.onerror = function(error) {
        reject(error);
      };
    });
  }

  handleImageSelect = (index, image) => {
    // Call the onSelect prop passed from App.js
    if (this.props.onSelect) {
      this.props.onSelect(index, image);
    }
  };

  render() {
    return <Gallery images={this.state.images} onClick={this.handleImageSelect} />
  }
}

export { GridGallery, getImages };