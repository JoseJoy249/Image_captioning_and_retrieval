# Image captioning and retreival
This project involves the code needed to build image captioning networks that can be used as a backbone to design an 
Image retrieval system that recommends images that are similar to the input image. Similarity is measured using Jaccard
similarity metric which uses captions (high level text representation of the image).

The original code for image captioning entwork was taken from following the repository [pytorch-tutorial/image-captioning](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/README.md)

This code was tweeked to generate multiple encoder-decoder models

The image retreival system uses COCO 2014 training and validation datasets and our own dataset scraped from Google Images (Proof of concept), from which the images are recommended.

All pretrained models can be accessed from [this link ](https://drive.google.com/drive/folders/1PsAwLMprM7lnWdrzq1PkQVP40A-a8d3s?ogsrc=32)

#### Model1
This model was pretrained (by authors of the above mentioned repository). The encoder architecture is Resnet 152, decoder architecture was single layer LSTM with 512 dimensional hidden states. Each word is represented using 256 dimensional learnable embedding.

#### Model2
The encoder architecture is Resnet 101, decoder architecture is a single layer LSTM with 512 dimensional hidden states. Each word is represented using 256 dimensional learnable embedding.

#### Model3
The encoder architecture is Resnet 152, decoder architecture is a single layer LSTM with 256 dimensional hidden states. Each word is represented using 128 dimensional learnable embedding.

#### Model4
The encoder architecture is Resnet 101, decoder architecture is a double layer LSTM with 512 dimensional hidden states for both layers. Each word is represented using 256 dimensional learnable embedding.

#### Model5
The encoder architecture is Resnet 101, decoder architecture is a single layer LSTM with 512 dimensional hidden states for both layers. Each word is represented using 300 dimensional Glove Vectors which was kept fixed throughout training.

## Usage

#### 1. Install pycocotools

pip install pycocotools
     
#### 2. Clone repository

git clone https://github.com/JoseJoy249/Image_captioning_and_retrieval.git

#### 3. Download COCO 2014 training and validation datasets

      pip install -r requirements.txt 
      chmod +x download.sh
      ./download.sh
    

#### 4. Try running image_crapper.ipynb to create a dataset from google image, that also downloads captions

#### 5. Run image_retrieval_system.ipynb

 learned.

#### Model5
This model was trained from scratch. The encoder architecture is Resnet 101, decoder architecture was single layer LSTM with 512 dimensional hidden states for both layers. Each word was represented using 300 dimensional  Glove Vectors which was kept fixed throughout training.


