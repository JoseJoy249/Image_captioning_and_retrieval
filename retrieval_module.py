import torch
import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import os
import glob
import cv2

from torchvision import transforms 
from build_vocab import Vocabulary
from PIL import Image
from pycocotools.coco import COCO

import model_files as models
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def load_model(model, device, vocab):
    """function to load trained encoder and decoder models 
    Inputs
    model : if 1 loads model 1, 2 loads  model 2 ... till model 5
    device : cpu or cuda device
    vocab : vocabulary object
    """
    if model == 1:
        en_path1 = "./models/encoder-5-3000.pkl"
        de_path1 = "./models/decoder-5-3000.pkl"
        encoder1 = models.EN1(256).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder1 = models.DN1(256, 512, len(vocab), 1)
        encoder1 = encoder1.to(device)
        decoder1 = decoder1.to(device)
        encoder1.load_state_dict(torch.load( en_path1 ))
        decoder1.load_state_dict(torch.load( de_path1 ))
        return encoder1, decoder1
        
    elif model == 2:
        en_path2 = "./models/encoder-arch1-5-3000.ckpt"
        de_path2 = "./models/decoder-arch1-5-3000.ckpt"
        encoder2 = models.EN2(256).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder2 = models.DN2(256, 512, len(vocab), 1)
        encoder2 = encoder2.to(device)
        decoder2 = decoder2.to(device)
        encoder2.load_state_dict(torch.load( en_path2 ))
        decoder2.load_state_dict(torch.load( de_path2 ))
        return encoder2, decoder2
        
    elif model == 3:
        en_path3 = "./models/encoder-arch2-4-3000.ckpt"
        de_path3 = "./models/decoder-arch2-4-3000.ckpt"
        encoder3 = models.EN3(128).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder3 = models.DN3(128, 256, len(vocab), 1)
        encoder3 = encoder3.to(device)
        decoder3 = decoder3.to(device)
        encoder3.load_state_dict(torch.load( en_path3 ))
        decoder3.load_state_dict(torch.load( de_path3 ))
        return encoder3, decoder3
    
    elif model == 4:
        en_path4 = "./models/encoder-arch3-5-3000.ckpt"
        de_path4 = "./models/decoder-arch3-5-3000.ckpt"
        encoder4 = models.EN4(256).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder4 = models.DN4(256, 512, len(vocab), 2)
        encoder4 = encoder4.to(device)
        decoder4 = decoder4.to(device)
        encoder4.load_state_dict(torch.load( en_path4 ))
        decoder4.load_state_dict(torch.load( de_path4 ))
        return encoder4, decoder4
        
    elif model == 5:
        # load weights_matrix for model5, glove vectors
        with open("./data/weights_matrix.pkl", "rb") as f:
            weights_matrix = torch.tensor( pickle.load(f) )
    
        en_path5 = "./models/encoder-arch4-5-3000.ckpt"
        de_path5 = "./models/decoder-arch4-5-3000.ckpt"
        encoder5 = models.EN5(300).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder5 = models.DN5(300, 512, len(vocab), 1, weights_matrix)
        encoder5 = encoder5.to(device)
        decoder5 = decoder5.to(device)
        encoder5.load_state_dict(torch.load( en_path5 ))
        decoder5.load_state_dict(torch.load( de_path5 ))
        return encoder5, decoder5
        
    else:
        print ("Incorrect input")
        
def get_COCO_caption_dict(dataset):
    """function to get caption dictionary, which has captions as keys and image_paths as values
    Inputs
    dataset : either train2014, val2014 or google_images
    """
    caption_dict = {}
    if dataset == "train2014":
        coco = COCO("./data/annotations/captions_train2014.json")
        id_list = list(coco.anns.keys())
        for ann_id in id_list:
            caption = coco.anns[ann_id]['caption']
            img_id = coco.anns[ann_id]['image_id']
            image_path = coco.loadImgs(img_id)[0]['file_name']
            caption_dict[caption] = image_path
            
    elif dataset == "val2014":
        coco = COCO("./data/annotations/captions_val2014.json")
        id_list = list(coco.anns.keys())
        for ann_id in id_list:
            caption = coco.anns[ann_id]['caption']
            img_id = coco.anns[ann_id]['image_id']
            image_path = coco.loadImgs(img_id)[0]['file_name']
            caption_dict[caption] = image_path
            
    return coco, caption_dict

def load_image(image_path):
    """function to load images from a path, and transforming it"""
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
        
    return image

def get_captions(image_path, encoder, decoder, vocab, device, beam_width = 1):
    """function to generate captions from an image using the model
    Inputs
    image_path = input image path
    encoder = encoder CNN model
    decoder = encoder CNN model
    vocab  = vocabulary datastructure
    device = GPU/CPU device
    beam_width = beam width search parameter. if 1, greedy search
    """
    
    image = load_image(image_path)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    if beam_width == 1:
        sampled_ids = decoder.sample(feature)
    else:
        sampled_ids = decoder.sample_beam(feature,  vocab.word2idx["<end>"], beam_width )
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
            
    return ' '.join(sampled_caption[1:-1])

    
def Jaccard_similarity(sen1, sen2):
    """function to find Jaccard similarity between two sentences sen1, sen2"""
    sen1 = sen1.lower().split(" ")
    sen2 = sen2.lower().split(" ")
    sen1 = set(sen1) 
    sen2 = set(sen2)
    inter = sen1 & sen2
    union = sen1 | sen2
    return len(inter)/len(union)
        
    
def retrieval_caption(caption, caption_dict, dataset, num = 5):
    """function to implement image retreival. It returns images similar to given caption
    Inputs
    caption = description of image
    caption_dict = dictionary whose keys are captions, values are annotation_id, image_id and image path
    dataset = image dataset (train2014, val2014, google_images)
    num = number of results to be displayed"""
    
    match = []
    for cap in caption_dict:
        score = Jaccard_similarity(caption, cap )
        match.append([score, caption_dict[cap]])
        
    match = sorted(match, key = lambda x:x[0])[-2*num:]
    
    if dataset == "train2014":
        temp_path = "./data/train2014/"
    elif dataset == "val2014":
        temp_path = "./data/val2014/"
    elif dataset == "google_images":
        temp_path = "./data/google_images/"
    
    print("Images similar to caption : ", caption)
    c = 0
    plt.figure(figsize= (15,15))
    
    for m in match:
        path1 =  temp_path + m[1]
        c += 1
        image = np.asarray( Image.open(path1) )
        plt.subplot(1,num,c)
        plt.axis("off")
        plt.imshow( cv2.resize(image, (256,256) ))

        if c==num:
            plt.show()
            return
    return
    
def retrieval_image(image_path, caption_dict, encoder, decoder, vocab, device, dataset, num = 5):
    """function to implement image retreival. It returns images similar to input image, based on caption
    Inputs
    im_path = path to input image
    caption_dict = dictionary whose keys are captions, values are annotation_id, image_id and image path
    dataset = train, val or google_images
    num = number of results to be displayed"""
    
    sentence = get_captions(image_path, encoder, decoder, vocab, device)
    
    match = []
    for cap in caption_dict:
        score = Jaccard_similarity(sentence, cap )
        match.append([score, caption_dict[cap]])
        
    match = sorted(match, key = lambda x:x[0])[-2*num:]
    
    if dataset == "train2014":
        temp_path = "./data/train2014/"
    elif dataset == "val2014":
        temp_path = "./data/val2014/"
    elif dataset == "google_images":
        temp_path = "./data/google_images/"

    
    print("\t Predicted caption : ",sentence)
    print("\t Recommendations : ")
    c = 0
    plt.figure(figsize= (15,15))
    for m in match:
        path1 = temp_path + m[1]
        if path1 != image_path:
            c += 1
            image = np.asarray( Image.open(path1) )
            plt.subplot(1,num,c)
            plt.axis("off")
            plt.imshow( cv2.resize(image, (256,256) ))
            
        if c==num:
            plt.show()
            return
    
    



