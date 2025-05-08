import os
import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import model
import torchvision

batch_size = 4
epochs = 1000

train_data = torchvision.datasets.ImageFolder('/home/data3t/panxuhao/Gan_train', \
                                                utils.TinyImageNetPairTransform())

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generator = model.Generator().to(device)
discriminator = model.Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters())
d_optimizer = optim.Adam(discriminator.parameters())

adversarial_loss = torch.nn.BCELoss()  
content_loss = torch.nn.L1Loss()                 

for epoch in range(1, epochs + 1):
    total_max_loss, total_num, train_bar = 999, 0, tqdm(train_loader)
    for data_tuple in train_bar:
        (clean_imgs, noisy_imgs), _ = data_tuple

        noisy_imgs = noisy_imgs.to(device)  
        clean_imgs = clean_imgs.to(device) 

        #train disc
        d_optimizer.zero_grad()
        
        real_pred = discriminator(clean_imgs)
        d_loss_real = adversarial_loss(real_pred, torch.ones_like(real_pred).to(device))

        generated_imgs = generator(noisy_imgs).detach()
        fake_pred = discriminator(generated_imgs)
        d_loss_fake = adversarial_loss(fake_pred,torch.zeros_like(fake_pred).to(device))

        d_total_loss = (d_loss_real + d_loss_fake) / 2
        d_total_loss.backward()
        d_optimizer.step()

        #train gen
        g_optimizer.zero_grad()

        generated_imgs = generator(noisy_imgs)

        gen_pred = discriminator(generated_imgs)
        g_loss_adv = adversarial_loss(gen_pred,torch.ones_like(gen_pred).to(device))

        g_loss_content = content_loss(generated_imgs, clean_imgs)

        g_total_loss = g_loss_adv + 100 * g_loss_content
        g_total_loss.backward()
        g_optimizer.step()

        print("Epoch:",epoch," loss_gen:",g_total_loss.item(), " loss_dis:",d_total_loss.item())

        if g_total_loss.item() < total_max_loss:
            total_max_loss = g_total_loss.item()
            torch.save(generator.state_dict(),"/home/data3t/panxuhao/GAN-main/GenG2.pt")
