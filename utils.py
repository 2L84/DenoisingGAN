from PIL import Image
from torchvision import transforms
import torch
import numpy as np

class TinyImageNetPairTransform:
    def salt_and_pepper(self, img, Nd=0.1):
        img = np.array(img)                                                        
        h, w, c = img.shape
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      
        mask = np.repeat(mask, c, axis=2)                                             
        img[mask == 0] = 0                                                            
        img[mask == 1] = 255                                                           
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                       
        return img
    
    def Guas(self, img, std=0.1, mean=0.0):
        noisy = torch.randn(img.size())*std + mean
        return img + noisy

    def __init__(self, ):
        self.resize = transforms.Compose([
            transforms.Resize(size=[256,256])
            ])
        self.tot = transforms.Compose([
            transforms.ToTensor()
            ])

    def __call__(self, x):
        noise = x
        noise = self.resize(noise)
        x = self.resize(x)
        noise = self.salt_and_pepper(noise)
        return self.tot(x), self.tot(noise)
