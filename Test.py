import time
import gc
import torch
import math
import os
import albumentations as A
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
# from torchsummary import summary
# from torchmetrics import PeakSignalNoiseRatio
from skimage import color
from skimage.metrics import structural_similarity

from Utils import load_checkpoint
from Dataset import ABDataset
from Model import Generator

gc.collect()
torch.cuda.empty_cache()

logdir = '08071145-trainval'
checkpoint = f"output/{logdir}/best_epoch/genb.pth.tar"
save_path = f"output/{logdir}/test_img"
TEST_DIR = f"../CycleGAN/test_datasets"

if not os.path.exists(save_path):
    os.mkdir(save_path)

def masking(a, b):
    l_top = l_bottom = 0
    a = a[0]
    b = b[0]

    for i in range(a.shape[1]):
        if torch.sum(a[:, i, :]) != 0:
            break
        l_top += 1

    for i in range(a.shape[1]):
        if torch.sum(a[:, a.shape[1] - i - 1, :]) != 0:
            break
        l_bottom += 1

    b[:, :l_top, :] = 0
    b[:, b.shape[1] - l_bottom:, :] = 0

    return a, b


def PSNR_SSIM(orig_img, gen_img):
    gray_orig_img = color.rgb2gray(orig_img)
    gray_gen_img = color.rgb2gray(gen_img)

    mse = np.mean((gray_orig_img - gray_gen_img) ** 2)
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    ssim = structural_similarity(gray_orig_img, gray_gen_img, multichannel=False, data_range=1.0)

    return round(psnr, 3), round(ssim, 3)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator().to(DEVICE)
# summary(gen, (3, 256, 256))

load_checkpoint(checkpoint, gen, None, None)

transforms_ = [
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

val_dataset = ABDataset(
    root_a=TEST_DIR, transforms_=transforms_
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
)

loop = tqdm(val_loader, leave=True)
psnr_values = []
ssim_values = []

start = time.time()

for idx, (image, filename) in enumerate(loop):
    image = image.to(DEVICE)

    with torch.cuda.amp.autocast():
        gen_image = gen(image)
        image, gen_image = masking(image*0.5+0.5, gen_image*0.5+0.5)
        save_image(gen_image, f"{save_path}/{filename[0]}")
        # save_image(image, f"{save_path}/{idx}_real.png")

        image = image.permute(1, 2, 0).detach().cpu().numpy()
        gen_image = gen_image.permute(1, 2, 0).detach().cpu().numpy()

        psnr_values.append(PSNR_SSIM(image, gen_image)[0])
        ssim_values.append(PSNR_SSIM(image, gen_image)[1])

end = time.time()

metrics = [
    round(sum(psnr_values) / len(val_loader), 3),
    round(sum(ssim_values) / len(val_loader), 3),
    round((end - start) / len(val_loader), 3)
]

print(f"Testing PSNR :{metrics[0]} dB\n")
print(f"Testing SSIM :{metrics[1]}\n")
print(f"Single image time: {metrics[2]} seconds\n")
