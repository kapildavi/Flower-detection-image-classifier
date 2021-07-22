# -*- coding: utf-8 -*-
# Imports here
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import argparse

from PIL import Image

from collections import OrderedDict

import time

import numpy as np
import matplotlib.pyplot as plt

from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()



def validation(model,criterion, testloader):
    device = torch.device('cuda')
    model.cuda();
    testloss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        testloss += criterion(output,labels).item()

        ps = torch.exp(output)
        equality = (labels.data ==ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return testloss, accuracy

def train(model, criterion, optimizer, trainloader, epochs,testloader):
    device = torch.device('cuda')
    model.cuda();
    import time
    epochs = 3
    print_every = 10
    steps = 0
    running_loss = 0
    
    stp=[]
    ts_loss = []
    tr_loss = []
    accu = []
    #total_time = time.time()
    for e in range(epochs):
        model.train()
        
        for images,labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model.forward(images)
            loss = criterion(logits, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    testloss, accuracy = validation(model,criterion, testloader)
                    
                    
                print('Epochs {}/{} '.format(e+1, epochs),
                      'train loss {:.3f} '.format(running_loss/print_every),
                      'test loss {:.3f} '.format(testloss/len(testloader)),
                      'accuracy {:.3f} '.format(accuracy/len(testloader)))
                
                tr_loss.append(running_loss/print_every)     
                stp.append(steps)    
                ts_loss.append(testloss/len(testloader))
                accu.append(accuracy/len(testloader))
                
                running_loss = 0
                model.train()
    
    #print("total_time {} minutes".format(time.time()-total_time())/60) 
    
    
def main():
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    train_set = datasets.ImageFolder(train_dir, transform = train_transform)
    test_set = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=16, shuffle=True)
    
    
    model = models.vgg16(pretrained=True)
    
    
    device = torch.device('cuda')
    
    for params in model.parameters() :
        params.requires_grad = False
        
    from collections import OrderedDict
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088,500)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p = 0.2)),
        ('fc2', nn.Linear(500,102)),
        ('output', nn.LogSoftmax(dim=1))    
        
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = train_set.class_to_idx
    gpu = args.gpu # get the gpu settings
    train(model, criterion, optimizer, trainloader, epochs,testloader)
    model.class_to_idx = class_index
    path = args.save_dir # get the new save location 
    save_checkpoint(path, model, optimizer, args, classifier)

    
if __name__ == "__main__":
    main()
    
    
    
    