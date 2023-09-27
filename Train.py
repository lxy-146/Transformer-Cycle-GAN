import os.path
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
# from torchsummary import summary

from Utils import save_checkpoint, load_checkpoint
from Dataset import ABDataset
from Model import Generator, Discriminator
from val import lxy_ssim_psnr
from logger import get_logger


import gc
gc.collect()
torch.cuda.empty_cache()

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

def train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse,  d_scaler, g_scaler, epoch):
    global count
    avg_dloss = 0
    avg_gloss = 0
    loop = tqdm(loader, leave=True)
    for idx, (a, b) in enumerate(loop):
        a = a.to(DEVICE)
        b = b.to(DEVICE)

        with torch.cuda.amp.autocast():
            fake_a = gen_A(b)
            D_A_real = disc_A(a)
            D_A_fake = disc_A(fake_a.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_b = gen_B(a)
            D_B_real = disc_B(b)
            D_B_fake = disc_B(fake_b.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            D_loss = (D_A_loss + D_B_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_A_fake = disc_A(fake_a)
            D_B_fake = disc_B(fake_b)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            cycle_b = gen_B(fake_a)
            cycle_a = gen_A(fake_b)
            cycle_b_loss = l1(b, cycle_b)
            cycle_a_loss = l1(a, cycle_a)

            identity_b = gen_B(b)
            identity_a = gen_A(a)
            identity_b_loss = l1(b, identity_b)
            identity_a_loss = l1(a, identity_a)

            G_loss = (
                loss_G_B
                + loss_G_A
                + cycle_b_loss * LAMBDA_CYCLE
                + cycle_a_loss * LAMBDA_CYCLE
                + identity_a_loss * LAMBDA_IDENTITY
                + identity_b_loss * LAMBDA_IDENTITY
            )

            avg_dloss += D_loss.item()
            avg_gloss += G_loss.item()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            save_image(fake_a*0.5+0.5, f"{path}/GeneratedfromHQ/{count}_fake.png")
            save_image(fake_b*0.5+0.5, f"{path}/GeneratedfromLQ/{count}_fake.png")
            save_image(b*0.5+0.5, f"{path}/GeneratedfromHQ/{count}_real.png")
            save_image(a*0.5+0.5, f"{path}/GeneratedfromLQ/{count}_real.png")
            count += 1
        loop.set_postfix(epoch=epoch+1, loss_g=avg_gloss/(idx+1), loss_d=avg_dloss/(idx+1))

def val(dataloader, gen):
    loop = tqdm(dataloader, leave=True)
    a_imgs = []
    b_imgs = []
    gen.eval()
    for idx, (a, b) in enumerate(loop):
        a = a.to(DEVICE)
        b = b.to(DEVICE)
        with torch.cuda.amp.autocast():
            gen_image = gen(a)
            a, gen_image = masking(a * 0.5 + 0.5, gen_image * 0.5 + 0.5)
            b = b * 0.5 + 0.5
            # save_image(gen_image, f"{save_path}/{filename[0]}")
            # save_image(image, f"{save_path}/{idx}_real.png")

            b = b.detach().cpu().numpy()
            gen_image = gen_image.detach().cpu().numpy()
            gen_image = gen_image * 255.0
            b = b * 255.0
            b = np.mean(b[0], axis=0)
            gen_image = np.mean(gen_image, axis=0)
            a_imgs.append(gen_image)
            b_imgs.append(b)
    ssim, psnr = lxy_ssim_psnr(a_imgs, b_imgs)
    gen.train()
    return ssim, psnr



def main():
    disc_A = Discriminator().to(DEVICE)
    disc_B = Discriminator().to(DEVICE)
    gen_A = Generator().to(DEVICE)
    gen_B = Generator().to(DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN_A, gen_A, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GEN_B, gen_B, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_A, disc_A, opt_disc, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_B, disc_B, opt_disc, LEARNING_RATE,
        )

    dataset = ABDataset(
        root_a=TRAIN_DIR+"/A", root_b=TRAIN_DIR+"/B", transforms_=transforms_
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    valdataset = ABDataset(root_a=TEST_DIR + '/A', root_b=TEST_DIR + '/B', transforms_=transforms_val)
    valloader = DataLoader(dataset,
                           batch_size=1,
                           shuffle=False,
                           pin_memory=True,)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    best_ssim, best_psnr = -1, -1
    for epoch in range(NUM_EPOCHS):
        train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

        if epoch % 10 == 0:
            ssim, psnr = val(valloader, gen_B)
            if ssim + 0.01 * psnr > best_ssim + 0.01 * best_psnr:
                best_ssim, best_psnr = ssim, psnr
                logger.info(f'Update best epoch:{epoch}, ssim:{ssim}, psnr:{psnr}')
                save_checkpoint(gen_B, opt_gen, filename=os.path.join(path, 'best_epoch', 'genb.pth.tar'))
        if SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=CHECKPOINT_DISC_A)
            save_checkpoint(disc_B, opt_disc, filename=CHECKPOINT_DISC_B)

def get_str_time(x):
    if x < 10:
        s = f'0{str(x)}'
    else:
        s = str(x)
    return s

def get_time_head():
    struct_time = time.localtime(time.time())
    time_head = get_str_time(struct_time.tm_mon) + get_str_time(struct_time.tm_mday) + get_str_time(
        struct_time.tm_hour) + get_str_time(struct_time.tm_min)
    return time_head

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
    parser.add_argument('--batchsize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='../CycleGAN/dataset/US', help='root directory of the dataset')
    parser.add_argument('--g_lr', type=float, default=1e-5, help='initial 0learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-5, help='initial 0learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--logdir', type=str, default='delete')
    parser.add_argument('--discriminator', type=str, default='discriminator', choices=['discriminator', 'sn'])
    parser.add_argument('--generator', type=str, default='generator', choices=['generator', 'vit'])
    parser.add_argument('--gpu', type=int, default=0)
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DIR = os.path.join(args.dataroot, 'train')
    TEST_DIR = os.path.join(args.dataroot, 'test')
    time_head = get_time_head()
    args.logdir = os.path.join('output', f'{time_head}-{args.logdir}')
    path = args.logdir
    BATCH_SIZE = args.batchsize
    LEARNING_RATE = args.g_lr
    LAMBDA_IDENTITY = 10
    LAMBDA_CYCLE = 10
    NUM_WORKERS = 4
    NUM_EPOCHS = args.n_epochs
    LOAD_MODEL = False
    SAVE_MODEL = True
    logger = get_logger(path)
    logger.info(args)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, 'GeneratedfromHQ')):
        os.mkdir(os.path.join(path, 'GeneratedfromHQ'))
    if not os.path.exists(os.path.join(path, 'GeneratedfromLQ')):
        os.mkdir(os.path.join(path, 'GeneratedfromLQ'))
    if not os.path.exists(os.path.join(path, 'best_epoch')):
        os.mkdir(os.path.join(path, 'best_epoch'))
    CHECKPOINT_GEN_A = f"{path}/gena.pth.tar"
    CHECKPOINT_GEN_B = f"{path}/genb.pth.tar"
    CHECKPOINT_DISC_A = f"{path}/disca.pth.tar"
    CHECKPOINT_DISC_B = f"{path}/discb.pth.tar"
    count = 0
    # transforms = A.Compose(
    #     [
    #         # A.Resize(width=256, height=256),
    #         # A.HorizontalFlip(p=0.5),
    #         # A.VerticalFlip(p=0.5),
    #         A.Normalize(mean=[0.5], std=[0.5]),
    #         ToTensorV2(),
    #      ],
    #     additional_targets={"image0": "image"},
    # )
    transforms_ = [transforms.Resize(int(256 * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(256),
                   transforms.RandomHorizontalFlip(),
                   # transforms.AutoAugment(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transforms_val = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    # summary(gen, (3, 256, 256))
    # summary(disc, (3, 256, 256))

    main()
