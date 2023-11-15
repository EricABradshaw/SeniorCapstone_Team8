import React from 'react';
import { Gallery } from "react-grid-gallery";

const NUM_IMAGES_REQUEST_FROM_API = 33;

function getImages(numberOfImages) {
  const apiUrl = 'https://picsum.photos/224/224';

  return new Promise((resolve, reject) => {
    if (!Number.isInteger(numberOfImages) || numberOfImages <= 0) {
      reject(new Error('Invalid number of images'));
    }

    // Make a request to the image API
    const imagePromises = [];

    for (let i = 0; i < numberOfImages; i++) {
      imagePromises.push(`${apiUrl}?random=${i}`);
    }

    Promise.all(imagePromises)
      .then(imageUrls => {
        resolve(imageUrls);
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
      images: [],
    };
  }

  componentDidMount() {
     getImages(NUM_IMAGES_REQUEST_FROM_API)
      .then(data => {
        // Assuming data is an array of image objects with URLs
        let image_array = []

        data.forEach(element => {
          image_array.push({
            src: element,
            width: 224,
            height: 224,
          });
        });

        this.setState({ images: image_array });
      })
      .catch(error => {
        console.error(error.message);
      });

  }

  handleImageSelect = (index, image) => {
    // Call the onSelect prop passed from App.js
    if (this.props.onSelect) {
      this.props.onSelect(index, image);
    }
  };

  render() {

    return <Gallery images={this.state.images}
      onSelect={this.handleImageSelect} />
  }
}

export default GridGallery;