# Image captioning and retreival
This project involves the code needed to build image captioning networks that can be used as a backbone to design an 
Image retrieval system that recommends images that are similar to the input image. Similarity is measured using Jaccard
similarity metric which uses captions (high level text representation of the image).

The original code for image captioning entwork was taken from following the repository [pytorch-tutorial/image-captioning](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/README.md)

This code was tweeked to generate multiple encoder-decoder models

The image retreival system uses COCO 2014 training and validation datasets and our own dataset scraped from Google Images (Proof of concept), from which the images are recommended.

## Usage

#### 1. Install pycocotools

#### 2. Clone repository

#### 3. Download COCO 2014 training and validation datasets

#### 4. Try running image_crapper.ipynb to create a dataset from google image, that also downloads captions

#### 5. Run image_retrieval_system.ipynb



