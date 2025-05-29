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
parser.add_argument('--train_steps', default=200001, type=int)
parser.add_argument('--lr_rate', default=2e-5, help="learning rate")
parser.add_argument('--sample_every_iterations', default=1000, type=int, help="sample SR images for every number of iterations")
parser.add_argument('--save_folder', default="GDTLS_segment_special_noise_256", type=str, help="Folder to save your train or evaluation result")
parser.add_argument('--load_path', default=None, type=str, help="None or directory to pretrained model")

# parser.add_argument('--data_path', default='/hdda/Datasets/afhq/all_images/', type=str, help="directory to your training dataset")
parser.add_argument('--data_path', default='/hdda/Datasets/Face_super_resolution/images1024x1024/', type=str, help="directory to your training dataset")
parser.add_argument('--batch_size', default=16, type=int)

args = parser.parse_args()

# if args.interval_mode == "linear":
#     size_list = []
#     timestep = (args.hr_size - args.lr_size) // args.stride
#     for i in reversed(range(timestep+1)):
#         size_list.append(int(args.lr_size + i * args.stride ))
# elif args.interval_mode == "exp":
#     size_list = []
#     timestep = int(math.log2(args.hr_size) - math.log2(args.lr_size))
#     for i in reversed(range(timestep+1)):
#         size_list.append(int(2 ** (i+math.log2(args.lr_size))))
# elif args.interval_mode == "fibonacci":
#     size_list = [args.hr_size]
#     fibonacci_list = [args.lr_size]
#     n = 0
#     m = 1
#     nth = 0
#     while (fibonacci_list[-1] + nth) < args.hr_size:
#         nth = n + m
#         fibonacci_list.append(args.lr_size + nth)
#         n = m
#         m = nth
#     for i in reversed(fibonacci_list):
#         if i < args.hr_size // 3:
#             size_list.append(i)
#     print(size_list)
#     timestep = len(size_list) - 1

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
    trainer.train()

