import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize,Resize
from PIL import Image
import numpy as np
import os
import torchvision.transforms as tranforms
import functools


# 超参数
INPUT_SHAPE = (3, 256, 256)  # 输入图像的形状
OUTPUT_SHAPE = (3, 256, 256)  # 输出图像的形状
BATCH_SIZE = 1  # 批次大小
EPOCHS = 30  # 训练轮数
LR = 2e-4  # 学习率
PRINT_FREQ = 2
SAVE_FREQ = 2

# 文件路径
TRAIN_PATH = './datasets/train/'  # 训练集路径
VAL_PATH = './datasets/val/'  # 验证集路径
SAVE_PATH = './models/'  # 模型保存路径

#更复杂的生成器

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

# 定义生成器网络结构
# class Generator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(Generator, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

###更复杂的鉴别器###
class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
# 定义判别器网络结构
# class Discriminator(nn.Module):
#     def __init__(self, in_channels):
#         super(Discriminator, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels*2, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 1, 4, 1, 1),
#         )

#     def forward(self, x, y):
#         x = torch.cat([x, y], dim=1)
#         x = self.conv(x)
#         return x

# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class Discriminator(nn.Module):
#     def __init__(self, input_channels=3, hidden_dim=32, num_heads=4, num_layers=4, dropout=0.1):
#         super(Discriminator, self).__init__()

#         self.embedding = nn.Sequential(
#             nn.Conv2d(input_channels*2, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True)
#         )

#         encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.fc = nn.Linear(hidden_dim*64*64, 1)

#     def forward(self, x, y):
#         x = torch.cat([x, y], dim=1)
#         x = self.embedding(x)
#         x = x.flatten(2).permute(2, 0, 1)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 2, 0).flatten(1)
#         x = self.fc(x)
#         return x


# 定义数据读取和预处理
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = sorted(os.listdir(root))
        self.root = root


    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_A = transform(img.crop((0, 0, w//2, h)))
        img_B = transform(img.crop((w//2, 0, w, h)))
        return {'B': img_A, 'A': img_B}

    def __len__(self):
        return len(self.files)

train_dataset = ImageDataset(TRAIN_PATH)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ImageDataset(VAL_PATH)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

discriminator = Discriminator(input_nc=6).cuda()
generator = UnetGenerator(input_nc=3,output_nc=3,num_downs=7).cuda()


optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.L1Loss()

# 训练模型
for epoch in range(EPOCHS):
    for i, batch in enumerate(train_dataloader):
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()

        # 训练判别器
        optimizer_D.zero_grad()

        fake_B = generator(real_A)
        pred_real = discriminator(torch.cat((real_A, real_B),1))
        pred_fake = discriminator(torch.cat((real_A, fake_B),1))
        loss_D = (criterion(pred_real, torch.ones_like(pred_real).cuda()) +
                  criterion(pred_fake, torch.zeros_like(pred_fake).cuda()))
        loss_D.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()

        fake_B = generator(real_A)
        pred_fake = discriminator(torch.cat((real_A, fake_B),1))
        loss_G = criterion(pred_fake, torch.ones_like(pred_fake).cuda())

        loss_G.backward()
        optimizer_G.step()

        # 打印训练进度
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % PRINT_FREQ == 0:
            val_losses = []
            for j, val_batch in enumerate(val_dataloader):
                real_A = val_batch['A'].cuda()
                real_B = val_batch['B'].cuda()

                with torch.no_grad():
                    fake_B = generator(real_A)



                    # 假设 real_A 是一个4维的张量，大小为 (batch_size, channels, height, width)
                    Temp = fake_B.shape[0]
                    transform_temp = tranforms.ToPILImage()

                    for i in range(Temp):
                        # 选择其中一张图片进行处理
                        img = fake_B[i]
                        # # 将通道维度移到最后
                        # img = img.permute(1, 2, 0)
                        # 转换成PIL Image
                        img_pil = transform_temp(img)
                        # img_pil.show()
                        # 保存图片
                        img_pil.save("./discri_Result/"+f"image_discri_epoch_{epoch}_{i}.png")
                        break

                    
                    #将Tensor变为Image
                    # transform = tranforms.ToPILImage()
                    # img = transform(real_A)
                    # img.show()

                    pred_real = discriminator(torch.cat((real_A, real_B),1))
                    pred_fake = discriminator(torch.cat((real_A, fake_B),1))

                val_loss_D = (criterion(pred_real, torch.ones_like(pred_real).cuda()) +
                              criterion(pred_fake, torch.zeros_like(pred_fake).cuda()))

                val_loss_G = criterion(pred_fake, torch.ones_like(pred_fake).cuda())

                val_losses.append((val_loss_D.item(), val_loss_G.item()))

            avg_val_loss_D = sum([l[0] for l in val_losses]) / len(val_losses)
            avg_val_loss_G = sum([l[1] for l in val_losses]) / len(val_losses)

            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(train_dataloader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                  f"[Val D loss: {avg_val_loss_D:.4f}] [Val G loss: {avg_val_loss_G:.4f}]")

        # 保存生成器的checkpoint
        if batches_done % SAVE_FREQ == 0:
            torch.save(generator.state_dict(), f"{SAVE_PATH}/generator_dis_{batches_done}.pth")
