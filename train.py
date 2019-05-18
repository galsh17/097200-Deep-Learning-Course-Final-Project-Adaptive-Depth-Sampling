import PIL.Image
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import copy
import time
import datetime
from utils import *
from model_Unet import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_path_test = '/content/gdrive/My Drive/deep_learning_project/test/'

hyper =  {
    "numEpochs": 250,
    "lr_initial": 4e-3 ,
    "lr_final": 5e-4 ,
    "cos_cycle": 30,
    "momentum": 0.9,
    "dropOut": 0,
    "batchSize": 10,
    "weight_decay": 0,
}

class depthDs(Dataset):
    #construct the dataset of rgb+d+sampling images
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
      #open the files using the csv file
        rgb_name = os.path.join(self.root_dir, self.files['rgb'][idx])
        sampling_name = os.path.join(self.root_dir, self.files['gt'][idx])
        depth_name = os.path.join(self.root_dir, self.files['depth'][idx])
        
        rgb_img = Image.open(rgb_name)
        depth_img = Image.open(depth_name)
        sampling_img = Image.open(sampling_name)
        
        sampling_img_gauss = sampling_img.convert("L").filter(ImageFilter.GaussianBlur(2))
        sampling_img_gauss = sampling_img_gauss / np.amax(np.array(sampling_img_gauss))
        sampling_img_gauss = Image.fromarray(sampling_img_gauss)

        #apply transforms on depth, rgb and sampling image if needed
        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)
            sampling_img = self.transform(sampling_img)
            sampling_img_gauss = self.transform(sampling_img_gauss)
            
        return (rgb_img, depth_img, sampling_img, sampling_img_gauss)
    

#consider adding color jitter, brightness contrast ...
ds_train = depthDs(csv_file=base_path_test + 'samples_train.csv',
                                    root_dir=base_path_test,transform = transforms.Compose([
                                            #transforms.RandomCrop(224),
#                                             transforms.RandomHorizontalFlip(),
#                                             transforms.RandomVerticalFlip(),
                                            #transforms.RandomRotation(90),
                                            transforms.ToTensor(),
                                            ]))

ds_test = depthDs(csv_file=base_path_test + 'samples_test.csv',
                                    root_dir=base_path_test,transform = transforms.Compose([ 
                                            #transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            ]))

batch_s = hyper["batchSize"]
dataloader_train = DataLoader(ds_train, batch_size=batch_s, shuffle=True) 
dataloader_test = DataLoader(ds_test, batch_size=batch_s, shuffle=True) 

model = UNet(n_channels=3,n_classes=1) #only one class - high prob if sampling point, low prob if not
model = model.to(device)
model.train()
criterion = nn.BCELoss() 
criterion2 = nn.MSELoss() 
# optimizer = torch.optim.SGD(model.parameters(), lr=hyper["lr_initial"],
#                             momentum=hyper["momentum"], weight_decay=hyper["weight_decay"])

optimizer = torch.optim.Adam(model.parameters(),lr=hyper['lr_initial'])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,hyper["cos_cycle"],hyper["lr_final"])


num_epochs = hyper['numEpochs']
#initialize error and loss matrix for plotting later
curves = np.zeros([num_epochs,3])
lowest_error = 1000000
best_epoch = 0
#training
for epoch in range(num_epochs):
#     scheduler.step() #for decay
    #train
    ####################
    model.train()
    for i, (rgb_imgs, depth_imgs, sampling_imgs, sampling_imgs_gauss) in enumerate(dataloader_train):
        rgb_imgs = rgb_imgs.to(device)
        sampling_imgs = sampling_imgs.to(device)
        
#         scheduler.step()
        # Forward + Backword + Optimize
        
        optimizer.zero_grad()
        outputs = model(rgb_imgs)

        loss = criterion(outputs,sampling_imgs)
        loss.backward()
        optimizer.step()
#        print("epoch %d, batch %d" % (epoch, i))
    
    #calculate loss and error on the train and test sets, save in matrix for plotting later
    
    print("evaluating epoch %d" % (epoch))
    model.eval()
    train_AccLoss = calc_acc(dataloader_train,model)
    test_AccLoss = calc_acc(dataloader_test,model)
    curves[epoch,:] = [epoch,train_AccLoss,test_AccLoss]
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    print('Test Loss: %.4f' % (test_AccLoss))
    print('Train Loss: %.4f' % (train_AccLoss))
    #scheduler.step(train_AccLoss) #for reduce on plateu
    
    #experiment.log_metric("test error", test_AccLoss)
    #experiment.log_metric("train error", train_AccLoss)
    
    if (epoch+1)%10==0 and epoch>0:
      
      visualize_model(model,dataloader_test,num_images=4)
      
      fig1 = plt.figure(1)
      plt.semilogy(curves[1:epoch,0],curves[1:epoch,1], label='Train Loss')
      plt.semilogy(curves[1:epoch,0],curves[1:epoch,2], label='Test Loss')
      plt.legend(loc='upper right')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Train and Test Loss')
      plt.grid(True)
      plt.show()
      
#       scheduler.step() #for decay

      #if epoch>num_epochs-30:
        #experiment.log_figure(figure=fig1)

    #save model if best test accuracy acheived
    if test_AccLoss < lowest_error:
        best_model_wts = copy.deepcopy(model.state_dict())
        lowest_error = test_AccLoss
        best_epoch = epoch
    print('Best Test Loss So Far: %.4f At Epoch: %d' %(lowest_error,best_epoch))
    print( )
#load best weights        
model.load_state_dict(best_model_wts)    

date = datetime.datetime.now()
filename = 'Unet_{}-{}-{}'.format(date.day,date.month,date.year)
model.train()
torch.save(model.state_dict(), filename )
print(filename)
