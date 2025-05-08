import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import model
import torchvision

test_data = torchvision.datasets.ImageFolder('/home/data3t/panxuhao/Gan_test', \
                                                utils.TinyImageNetPairTransform())

test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)

Gen = model.Generator().cuda()
Gen.load_state_dict(torch.load("/home/data3t/panxuhao/GAN-main/GenG2.pt"))
Gen.eval()

test_bar = tqdm(test_loader)
i = 1
for data_tuple in test_bar:
    (origin, blur), _ = data_tuple
    origin, blur = origin.cuda(non_blocking=True), blur.cuda(non_blocking=True) 

    fake = Gen(blur)

    origin = origin.squeeze(dim=0)
    blur = blur.squeeze(dim=0)
    fake = fake.squeeze(dim=0)
    mask = fake - origin

    path = "/home/data3t/panxuhao/GAN-main"
    num = str(i)
    path = path+"/resultG2/"+num
    os.makedirs(path)

    toPIL = torchvision.transforms.ToPILImage()
    pic = toPIL(origin)
    pic.save(path+'/origin.jpg')
    pic = toPIL(blur)
    pic.save(path+'/blur.jpg')
    pic = toPIL(fake)
    pic.save(path+'/fake.jpg')
    pic = toPIL(mask)
    pic.save(path+'/mask.jpg')
    i += 1