import math
import torch.cuda
from util.models import *
from vdt import DTLS, Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:1", type=str)
parser.add_argument('--hr_size', default=256, type=int, help="size of HR image")
parser.add_argument('--lr_size', default=2, type=int, help="size of LR image")
parser.add_argument('--interval_mode', default="fibonacci", type=str, help="linear; exp; fibonacci")
parser.add_argument('--stride', default=2, type=int, help="size change between each step if linear mode is used")
parser.add_argument('--train_steps', default=5, type=int)
parser.add_argument('--lr_rate', default=2e-5, help="learning rate")
parser.add_argument('--sample_every_iterations', default=5000, type=int, help="sample SR images for every number of iterations")
parser.add_argument('--save_folder', default="Best_50k_fid", type=str, help="Folder to save your train or evaluation result")
parser.add_argument('--load_path', default="Training_results/GDTLS_segment_special_noise_256/GDTLS_200000.pt", type=str, help="None or directory to pretrained model")
# parser.add_argument('--load_path', default=None, type=str, help="None or directory to pretrained model")

parser.add_argument('--data_path', default='/hdda/Datasets/ffhq256/', type=str, help="directory to your training dataset")
# parser.add_argument('--data_path', default='testset', type=str, help="directory to your training dataset")
parser.add_argument('--fake_data_path', default='fake_dataset', type=str, help="directory to your training dataset")

parser.add_argument('--batch_size', default=1, type=int)

args = parser.parse_args()
device = args.device if torch.cuda.is_available() else "cpu"
size_list = [256, 64, 32, 16, 8, 6, 4, 3, 2]
timestep = len(size_list) - 1

print(f"Total steps for {args.lr_size} to {args.hr_size}: {timestep}")

model = UNet().to(device)
discriminator = Discriminator().to(device)

dtls = DTLS(
    model,
    image_size = args.hr_size,
    stride = args.stride,
    size_list=size_list,
    timesteps = timestep,        # number of steps
    device=device,
).to(device)


trainer = Trainer(
    dtls,
    discriminator,
    args.data_path,
    image_size = args.hr_size,
    train_batch_size = args.batch_size,
    train_num_steps = args.train_steps, # total training steps
    ema_decay = 0.995,                  # exponential moving average decay
    results_folder = args.save_folder,
    load_path = args.load_path,
    device = device,
    save_and_sample_every = args.sample_every_iterations
)

if __name__ == "__main__":
    trainer.evaluation()

