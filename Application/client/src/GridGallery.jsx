import React, {useEffect} from 'react';
import { Gallery } from "react-grid-gallery";

const images = [
   {
      src: "https://c2.staticflickr.com/9/8817/28973449265_07e3aa5d2e_b.jpg",
      width: 224,
      height: 224,
      caption: "After Rain (Jeshu John - designerspics.com)",
   },
   {
      src: "https://c2.staticflickr.com/9/8356/28897120681_3b2c0f43e0_b.jpg",
      width: 224,
      height: 224,
      tags: [
         { value: "Ocean", title: "Ocean" },
         { value: "People", title: "People" },
      ],
      alt: "Boats (Jeshu John - designerspics.com)",
   },
   {
      src: "https://c4.staticflickr.com/9/8887/28897124891_98c4fdd82b_b.jpg",
      width: 224,
      height: 224,
   },
];

class GridGallery extends React.Component {
  handleImageSelect = (index, image) => {
    // Call the onSelect prop passed from App.js
    if (this.props.onSelect) {
      this.props.onSelect(index, image);
    }
  };

  render() {
    return <Gallery images={images}
    onSelect={this.handleImageSelect} />
  }
}

export default GridGallery;