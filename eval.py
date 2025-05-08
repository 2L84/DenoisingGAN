from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import torch
imor=plt.imread('C:\\Users\\panxu\\Desktop\\o.png')
imnose=plt.imread('C:\\Users\\panxu\\Desktop\\n.png')
imde=plt.imread('C:\\Users\\panxu\\Desktop\\de.png')
t_o = torch.tensor(imor)
t_n = torch.tensor(imnose)
t_de = torch.tensor(imde)
print(t_n.size())
print(t_de.size())
print(t_o.size())
print('原图和噪点图的PSNR为{}'.format(PSNR(imor,imnose)))
print('原图和去噪图的PSNR为{}'.format(PSNR(imor,imde)))
print('原图和噪点图的SSIM为{}'.format(SSIM(imor,imnose,channel_axis=2,data_range=255)))
print('原图和去噪图的SSIM为{}'.format(SSIM(imor,imde,channel_axis=2,data_range=255)))
print('原图和噪点图的MSE为{}'.format(MSE(imor,imnose)))
print('原图和去噪图的MSE为{}'.format(MSE(imor,imde)))