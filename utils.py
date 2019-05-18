import torch
import matplotlib.pyplot as plt
import numpy as np

def metric_acc(preds,labels):
  with torch.no_grad():
    return criterion2(preds,labels)

def calc_acc(dataset,model):
    """Calculates model accuracy and loss on data set"""
    #TODO: add depth completion error later
    with torch.no_grad():
      tot = 0
      loss = 0
      model.eval()
      for i, (rgb_imgs, depth_imgs, sampling_imgs, sampling_imgs_gauss) in enumerate(dataset):
          model.eval()
          rgb_imgs = rgb_imgs.to(device)
          sampling_imgs_gauss = sampling_imgs_gauss.to(device)
          output = model(rgb_imgs)
          tot += sampling_imgs_gauss.size(0)
          loss += metric_acc(output,sampling_imgs_gauss)
      if i>0:
        loss /= i
    return loss.cpu().data.numpy()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if np.shape(inp)[2]==1: #if image is grayscale no need for last dim
      inp = np.squeeze(inp,2)
    plt.imshow(inp)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.grid(None)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

  
def visualize_model(model, dataloader, num_images=4): 
  
    was_training = model.training #check if model was in training or eval mode
    model.eval().cuda() 
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (rgb_imgs, depth_imgs, sampling_imgs, sampling_imgs_gauss) in enumerate(dataloader):
            outputs = model(rgb_imgs.cuda())
            preds = outputs.cpu()
            
            for j in range(rgb_imgs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                plt.xticks([], [])
                cat_sampling = torch.cat((sampling_imgs_gauss.cpu().data[j],outputs.cpu().data[j]),2)
                imshow(cat_sampling)
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
