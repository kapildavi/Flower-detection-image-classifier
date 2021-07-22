# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 18:42:53 2021

@author: admin
"""

import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F

import numpy as np

from PIL import Image

import json
import os
import random

from utils import load_checkpoint, load_cat_names
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/2/image_05100.jpg') # use a deafault filepath to a primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(path):
    image = Image.open(path)
    
    transformer = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    np_image = transformer(image).float()
    return np_image

def predict(path,model, topk =5):
    device = torch.device('cuda')
    img = process_image(path)
    img = img.float().unsqueeze_(0)
    img = img.to(device)
    with torch.no_grad():
        output = model.forward(img)
    ps = torch.exp(output)
    probs, indices = ps.topk(5)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    class_to_idx = {v:k for k,v in model.class_to_idx.items()}
    classes = [class_to_idx[x] for x in indices]
    return probs, classes

def main():
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    path = args.filepath
    probs, classes = predict(path, model, int(args.top_k))
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + path)
    
    print(labels)
    print(probability)
    
    i=0 # this prints out top k classes and probs as according to user 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 # cycle through

if __name__ == "__main__":
    main()