import sys
sys.path.append('./CV_homework/')
sys.argv = ['ipykernel_launcher.py']

import os, sys, argparse
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio

from src import *

mpl.rcParams['figure.figsize'] = (8, 8)

# Note: Actually it's totally unnecessay and silly to use argparse in jupyter notebook. 
# But anyway, it's a good chance to do some practice and it will definitey be useful someday.
parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=256, help="Image size")
parser.add_argument("--train_dataset", type=str, default="./afhq/train", help="The path to your training dataset")
parser.add_argument("--device", type=str, default=0 if torch.cuda.is_available() else "cpu", help="Device number.")
parser.add_argument("--num_workers", type=int, default=0, help="Spawn how many processes to load data.")
parser.add_argument("--seed", type=int, default=42, help='manual seed')
parser.add_argument("--max_epochs", type=int, default=1000, help="Max epoch number to run.")
parser.add_argument("--ckpt_path", type=str, default="./checkpoints/", help="Checkpoint path to load.")
parser.add_argument("--save_path", type=str, default="./checkpoints/", help="Checkpoint path to save.")
parser.add_argument("--save_freq", type=int, default=1, help="Save model every how many epochs.")
parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM timesteps")
# TODO begin: Add arguments lr and batch_size. It's recommended to set default lr to 1e-4 and default batch_size to 8.
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=8)

# TODO end DONE
args = parser.parse_args()
seed_everything(args.seed)

from kornia.utils import image_to_tensor
import kornia.augmentation as KA

class SimpleImageDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self,
                 root_dir,
                 transforms = None):

        self.root_dir = root_dir
        self.transforms = transforms

        # set up transforms
        if self.transforms is not None:
            data_keys = ['input']

            self.input_T = KA.container.AugmentationSequential(
                *self.transforms,
                data_keys = data_keys,
                same_on_batch = False
            )

        # TODO begin: Define the image paths filtered by the `supported_formats` in your datasets
        # Hint: os.listdir
        # Challenge: Can you complete this task in one line? (hint: Python comprehension, refer to Python basics handout by Yifan Li)
        supported_formats = ["jpg", "png"]
        self.image_names = [name for name in os.listdir(args.train_dataset) if name[-3:] in supported_formats]
        
        # TODO end
        # DONE

    def __len__(self):
        # TODO begin: Return the length of your dataset
        return len(self.image_names)
        # TODO end
        # DONE

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = image_to_tensor(imageio.imread(img_name)) / 255

        if self.transforms is not None:
            image = self.input_T(image)[0]

        return image
    

import torchvision.transforms as T

CROP_SIZE = args.image_size

transform = [
    KA.RandomCrop((2 * CROP_SIZE,2 * CROP_SIZE)),
    KA.Resize((CROP_SIZE, CROP_SIZE), antialias=True),
    KA.RandomVerticalFlip()
  ]

train_dataset = SimpleImageDataset(args.train_dataset, transforms = transform)

# TODO begin: Define the training dataloader using torch.utils.data.DataLoader
# Hint: check the API of torch.utils.data.DataLoader, especially arguments like batch_size, shuffle, num_workers
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)
# TODO end Done

# model training
from src import *

# TODO begin: complete the LatentDiffusion Model in `src`
model = LatentDiffusion(lr = args.lr, batch_size = args.batch_size)
# TODO end
# Done

img = train_dataset[0]

# TODO begin: Show the example img and use vae to reconstruct it using matplotlib
# Hint: plt.imshow
# Challenge: What's the image shape here? Should you permute or unsqueeze it?
# permute it, from [3,256,256] to [256,256,3]
plt.figure()
plt.subplot(1,2,1)

# Plot the original img here
img_plot = img.permute(1,2,0)
plt.imshow(img_plot)

plt.title('Input')

plt.subplot(1,2,2)

# Plot the reconstructed img by `model.vae` here
encoded_img = model.vae.encode(img.unsqueeze(0)) # torch.Size([1, 4, 32, 32])
reconstructed_img = model.vae.decode(encoded_img).squeeze(0).permute(1,2,0)
plt.imshow(reconstructed_img)

plt.title('AutoEncoder Reconstruction')

plt.savefig("test.png")
# TODO end
# Done


# Define the trainer using PyTorch Lightning
from pytorch_lightning.callbacks import ModelCheckpoint

# # You need to `pip install tensorboard` before importing
# from pytorch_lightning.loggers import TensorBoardLogger

# # 创建一个日志记录器，将日志保存在一个指定的目录中
# logger = TensorBoardLogger('logs/', name='model_logs')

# # 创建一个回调函数，用于记录训练指标
# class LoggingCallback(pl.Callback):
#     def on_epoch_end(self, trainer, pl_module):
#         # 从训练器中获取训练指标
#         train_loss = trainer.callback_metrics['train_loss']
#         valid_loss = trainer.callback_metrics['val_loss']

#         # 使用日志记录器将指标写入日志文件
#         logger.log_metrics({'train_loss': train_loss, 'valid_loss': valid_loss})

checkpoint_callback = ModelCheckpoint(dirpath=args.save_path, every_n_epochs=args.save_freq)

# TODO: You can specify other parameters here, like accelerator, devices...
# You can check the pl.Trainer API here: https://lightning.ai/docs/pytorch/stable/common/trainer.html
trainer = pl.Trainer(
    max_epochs = args.max_epochs,
    callbacks = [
        # LoggingCallback(), 
        EMA(0.9999), checkpoint_callback],
    # logger=logger,
    # , accelerator="gpu" # in fact, there is only cpu on the server
)
# Done
    

# Easy to train the model in PyTorch Lightning in one line
trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=args.ckpt_path if args.ckpt_path else None)

model.to(args.device)
out = model(batch_size = args.batch_size, shape = (64,64), verbose = True)

for idx in range(out.shape[0]):
    plt.subplot(1,len(out),idx+1)
    plt.imshow(out[idx].detach().cpu().permute(1,2,0))
    plt.axis('off')
    plt.savefig("./images/figure-{}.png".format(idx), dpi=600)