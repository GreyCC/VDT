import copy
import torch.nn.functional as F
import numpy as np
import glob
import shutil
import cv2
import os
import errno
import torch
import pyiqa
import shutil
import math
import wandb
import random

from torch import nn
from torch.utils import data
from pathlib import Path
from torch.optim import Adam, AdamW
from torchvision import transforms, utils
from PIL import Image

from util.fid_score import calculate_fid_given_paths

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

####### helpers functions

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DTLS(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        size_list,
        stride,
        timesteps,
        device,
        stochastic=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.UNet = model

        self.num_timesteps = int(timesteps)
        self.size_list = size_list
        self.stride = stride
        self.device = device
        self.MSE_loss = nn.MSELoss()


    def transform_func_sample(self, img, target_size):
        n = target_size
        m = self.image_size

        if m/n > 16:
            img_1 = F.interpolate(img, size=m//4, mode='bicubic', antialias=True)
            img_1 = F.interpolate(img_1, size=m//8, mode='bicubic', antialias=True)
            img_1 = F.interpolate(img_1, size=n, mode='bicubic', antialias=True)
        else:
            img_1 = F.interpolate(img, size=n, mode='bicubic', antialias=True)
        img_1 = F.interpolate(img_1, size=m, mode='bicubic', antialias=True)

        return  img_1

    def transform_func_noise(self, img, target_size, noise=None, random_mean=None, noise_level=256):
        n = target_size
        m = self.image_size

        if random_mean is None:
            random_mean = torch.rand(1).mul(2).add(-1).item()
        # sigma = (1 - math.log(n)/math.log(m)) * torch.rand(1).item() * sigma_ratio
        decreasing_scale = 0.9 ** (n - 2)


        if m/n > 16:
            img_1 = F.interpolate(img, size=m//4, mode='bicubic', antialias=True)
            img_1 = F.interpolate(img_1, size=m//8, mode='bicubic', antialias=True)
            img_1 = F.interpolate(img_1, size=n, mode='bicubic', antialias=True)
        else:
            img_1 = F.interpolate(img, size=n, mode='bicubic', antialias=True)
        # utils.save_image(img_1.add(1).mul(0.5), f"lr_{n}.png")

        if n <= noise_level:
            noise = torch.normal(mean=random_mean, std=0.5, size=(img_1.shape[0], 3, 2, 2)).to(self.device)
            # utils.save_image(noise.add(1).mul(0.5), f"noise_22_{n}.png")
            noise = F.interpolate(noise, size=n, mode='bicubic', antialias=True)
            # utils.save_image(noise.add(1).mul(0.5), f"noise_lr_{n}.png")
            img_1 += noise * decreasing_scale
            # utils.save_image(img_1.add(1).mul(0.5), f"lr_noised_{n}.png")

        img_1 = F.interpolate(img_1, size=m, mode='bicubic', antialias=True)
        # utils.save_image(img_1.add(1).mul(0.5), f"hr_noised_{n}.png")

        return  img_1


    def transform_func_noise_old(self, img, target_size, noise=None, random_mean=None, noise_level=256):
        n = target_size
        m = self.image_size

        random_scale = torch.rand(1).item()
        decreasing_scale = (0.8 ** (n - 2)) * random_scale


        if m/n > 16:
            img_1 = F.interpolate(img, size=m//4, mode='bicubic', antialias=True)
            img_1 = F.interpolate(img_1, size=m//8, mode='bicubic', antialias=True)
            img_1 = F.interpolate(img_1, size=n, mode='bicubic', antialias=True)
        else:
            img_1 = F.interpolate(img, size=n, mode='bicubic', antialias=True)

        if n <= noise_level:
            # noise = torch.normal(mean=random_mean, std=0.5, size=(img_1.shape[0], 3, 2, 2)).to(self.device)
            noise = torch.randn_like(img_1)
            # noise = F.interpolate(noise, size=n, mode='bicubic', antialias=True)
            img_1 += noise * decreasing_scale
            # utils.save_image(torch.cat((image_1_clone, noise.add(1).mul(0.5), img_1.add(1).mul(0.5)), dim=0), f"new_noise_test_{n}.png", nrow=noise.shape[0])

        img_1 = F.interpolate(img_1, size=m, mode='bicubic', antialias=True)

        return  img_1

    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None, save_folder=None):
        if t == None:
            t = self.num_timesteps

        blur_img = self.transform_func_sample(img.clone(), self.size_list[t])
        # random_mean = torch.rand(1).mul(2).add(-1).item()

        img_t = blur_img.clone()
        # import time
        # start_time = time.time()

        ####### Domain Transfer
        while (t):
            next_step = self.size_list[t-1]
            step = torch.full((batch_size,), t, dtype=torch.long).to(self.device)

            # utils.save_image(img_t.add(1).mul(0.5), f"intermediate_{t}.png")
            R_x = self.UNet(img_t, step)

            if t == 1:
                # print("--- %s seconds ---" % (time.time() - start_time))
                # utils.save_image(R_x.add(1).mul(0.5), f"intermediate_final.png")
                return blur_img, R_x
            else:
                img_t = self.transform_func_noise(R_x, next_step)

            t -= 1
            
        return blur_img, img_t
    
        
    def p_losses(self, x_start, t):
        x_blur = x_start.clone()
        #x_next = x_start.clone()

        for i in range(t.shape[0]):
            current_step = self.size_list[t[i]]
            x_blur[i] = self.transform_func_noise(x_blur[i].unsqueeze(0), current_step)

        x_recon = self.UNet(x_blur, t)

        ### Pattern Domain Similarity Loss
        x_clone = x_recon.clone()
        for i in range(t.shape[0]):
            current_step = self.size_list[t[i]]
            x_clone[i] = self.transform_func_sample(x_recon[i].unsqueeze(0), current_step)

        loss = self.MSE_loss(x_clone, x_blur)
        return loss, x_recon, t

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.1), int(image_size*1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        discriminator,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_num_steps = 100000,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        validate_every = 10000,
        results_folder,
        load_path = None,
        shuffle=True,
        device,
    ):
        super().__init__()
        
        ########## Wandb ##########
        wandb.init(project="GDTLS_256", notes=str(results_folder))
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.device = device
        
        self.model = diffusion_model
        self.discriminator = discriminator
        self.model_size()


        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.save_and_sample_every = save_and_sample_every
        self.validate_every = validate_every

        self.image_size = diffusion_model.image_size
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.nrow = 8

        self.folder_path = folder
        self.ds = Dataset(folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=2))

        self.opt = AdamW(diffusion_model.parameters(), lr=2e-5, betas=(0.0, 0.9), eps=1e-8)
        self.opt_d = AdamW(self.discriminator.parameters(), lr=4e-5, betas=(0.0, 0.9), eps=1e-8)

        self.BCE_loss = torch.nn.BCEWithLogitsLoss()

        self.step = 0

        self.reset_parameters()

        self.best_quality = 0
        self.load_path = load_path


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        line = ('model size: {:.3f}MB'.format(size_all_mb))
        print(line)
        wandb.config["model_size"] = size_all_mb

        # with open(f'{self.results_folder}/model_size.txt', 'w') as f:
        #     f.write('readme')

    def save_last(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'dis': self.discriminator.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'GDTLS_{self.step}.pt'))
    
    def save_best(self):
        data = {
            'step': self.step,
            'ema': self.ema_model.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'GDTLS_best.pt'))

    
    def load_all(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)
        self.discriminator.load_state_dict(data['dis'], strict=False)

    def load_for_eval(self, load_path):
        # print("Loading : ", "weight/gan.pt")
        data = torch.load(load_path, map_location=self.device)
        self.ema_model.load_state_dict(data['ema'], strict=False)



    def train(self):
        if self.load_path is not None:
            self.load_all(self.load_path)
        best_fid_score = 1e8

        while self.step < self.train_num_steps:
            data = next(self.dl) 
            data = data.to(self.device)
            
            loss_domain_sim, x_recon, t = self.model(data)
            loss_domain_sim = loss_domain_sim
            self.opt_d.zero_grad()
            score_true = self.discriminator(data)            
            GAN_true = torch.ones_like(score_true)
            loss_dis_true = self.BCE_loss(score_true, GAN_true)
            loss_dis_true.backward()

            score_false = self.discriminator(x_recon.detach())
            GAN_false = torch.zeros_like(score_false)
            loss_dis_false = self.BCE_loss(score_false, GAN_false)
            loss_dis_false.backward()

            self.opt_d.step()


            self.opt.zero_grad()
            score_fake = self.discriminator(x_recon)
            GAN_fake = torch.ones_like(score_fake)
            loss_gen = self.BCE_loss(score_fake, GAN_fake) * 1e-2
            
            (loss_gen + loss_domain_sim).backward()
            self.opt.step()

            if self.step % 10 == 0:
                print(f'{self.step} DTLS: Total loss: {loss_domain_sim.item() + loss_gen.item()} | Domain sim: {loss_domain_sim.item()} | Generate: {loss_gen.item()} '
                      f'| Dis real: {loss_dis_true.item()} | Dis false: {loss_dis_false.item()}')
            
            wandb.log({"Total loss": loss_domain_sim.item() + loss_gen.item(), "Domain Similarity Loss": loss_domain_sim.item(), 
                "Generation loss": loss_gen.item(), "Discriminator loss (real)": loss_dis_true.item(), "Discriminator loss (fake)": loss_dis_false.item()})

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step == 0 or self.step % self.save_and_sample_every == 0:
                lr_real, sr_real = self.ema_model.sample(batch_size=self.batch_size, img=data)
                save_img = torch.cat((lr_real, data, sr_real),dim=0)
                utils.save_image((save_img+1)/2, str(self.results_folder / f'{self.step}_GDTLS.png'), nrow=self.nrow)

                wandb.log({"Checkpoint result": wandb.Image(str(self.results_folder / f'{self.step}_GDTLS.png'))})

            if self.step % (10 * self.save_and_sample_every) == 0:
                self.save_last()

            # if self.step == 0 or self.step % self.validate_every == 0:
            #     create_folder(f"{self.results_folder}/temp_dataset")
                # self.validation(num_samples=1000, batch_size=self.batch_size)

                ### FID
                # fid_value = calculate_fid_given_paths([f"{self.folder_path}",
                #                                       f"{self.results_folder}/temp_dataset/"],
                #                                       device=self.device, batch_size=100, dims=2048)
                #
                # if fid_value < best_fid_score:
                #     best_fid_score = fid_value
                #     self.save_best()

                # del_folder(f"{self.results_folder}/temp_dataset")
                #
                # wandb.log({"FID score": fid_value})

            self.step += 1

        print('training completed')
        wandb.finish()


    def random_vector(self, batch_size):
        mean = random.uniform(-0.5, 0.5)
        std = random.uniform(0.1, 0.3)
        # print(f"mean: {mean} | std: {std}")
        return torch.normal(mean=mean, std=std, size=(batch_size, 3, 2, 2)).to(self.device)

    # def random_vector_rgb(self, batch_size):
    #     mean = random.uniform(-0.5, 0.5)
    #     std = random.uniform(0.1, 0.3)
    #     vector = torch.normal(mean=mean, std=std, size=(batch_size, 1, 2, 2))
    #     for i in range(2):
    #         mean = random.uniform(-0.5, 0.5)
    #         std = random.uniform(0.1, 0.3)
    #         rgb = torch.normal(mean=mean, std=std, size=(batch_size, 1, 2, 2))
    #         vector = torch.cat((vector,rgb), dim=1)
    #     return vector.to(self.device)

    def random_vector_rgb(self, batch_size):
        mean = random.uniform(-0.75, 0.75)
        std = random.uniform(0.01, 0.5)
        vector = torch.normal(mean=mean, std=std, size=(batch_size, 1, 2, 2))
        for i in range(2):
            mean = random.uniform(-0.75, 0.75)
            std = random.uniform(0.01, 0.5)
            rgb = torch.normal(mean=mean, std=std, size=(batch_size, 1, 2, 2))
            vector = torch.cat((vector, rgb), dim=1)
        return vector.to(self.device)

    def validation(self, num_samples=1, batch_size=1):
        total_img = 0
        for i in range((num_samples // batch_size + 1)):
            random_vector = self.random_vector_rgb(batch_size)
            lr, sample_hr = self.ema_model.sample(batch_size=batch_size, img=random_vector)
            for j in range(sample_hr.shape[0]):
                if total_img < num_samples:
                    utils.save_image((sample_hr[j] + 1) / 2, str(self.results_folder/"temp_dataset"/f'result_{total_img}.png'))
                    total_img += 1
                    print("saved: ", total_img)
        

    def evaluation(self, num_sample=50000):
        if self.load_path is not None:
            self.load_for_eval(self.load_path)
        # data = {
        #     'model': self.model.state_dict(),
        # }
        # torch.save(data, str(self.results_folder / f'GDTLS_UNet_only.pt'))

        for i in range(num_sample):
            # lr = next(self.dl)
            # lr = lr.to(self.device)
            random_vector = self.random_vector_rgb(1)
            _, sample_hr = self.ema_model.sample(batch_size=1, img=random_vector.to(self.device))
            print("saving: ", i)
            # random_vector = F.interpolate(random_vector, size=256, mode="nearest-exact")
            # utils.save_image((random_vector + 1) /2, str(self.results_folder /  f'random_vector_{i}.png'), nrow=4)
            utils.save_image((sample_hr + 1) /2, str(self.results_folder /  f'result_{i}.png'), nrow=1)

        # --------------------- Interpolate latent space ------------------- #
        # random_vector_1 = self.random_vector_rgb(1) * 0
        # _, sample_hr = self.ema_model.sample(batch_size=1, img=random_vector_1)
        # utils.save_image((sample_hr + 1) / 2, str(self.results_folder / f'result_1.png'), nrow=1)
        # a = F.interpolate(random_vector_1, size=256, mode = "nearest-exact")
        # utils.save_image((a + 1) / 2, str(self.results_folder / f'noise_1.png'), nrow=1)
        #
        # random_vector_2 = self.random_vector_rgb(1) * 2
        # _, sample_hr = self.ema_model.sample(batch_size=1, img=random_vector_2)
        # utils.save_image((sample_hr + 1) / 2, str(self.results_folder / f'result_2.png'), nrow=1)
        # a = F.interpolate(random_vector_2, size=256, mode="nearest-exact")
        # utils.save_image((a + 1) / 2, str(self.results_folder / f'noise_2.png'), nrow=1)
        #
        # for i in range(20):
        #     s = i + 1
        #     s /= 20
        #     print(s)
        #     new_vector = random_vector_1 * (1-s) + random_vector_2 * s
        #     _, sample_hr = self.ema_model.sample(batch_size=1, img=new_vector)
        #     print("saving: ", i)
        #     # random_vector = F.interpolate(random_vector, size=256, mode="nearest-exact")
        #     # utils.save_image((random_vector + 1) /2, str(self.results_folder /  f'random_vector_{i}.png'), nrow=4)
        #     utils.save_image((sample_hr + 1) /2, str(self.results_folder /  f'result_inter_{i}.png'), nrow=1)
        #     a = F.interpolate(new_vector, size=256, mode="nearest-exact")
        #     utils.save_image((a + 1) / 2, str(self.results_folder / f'noise_inter_{i}.png'), nrow=1)

        # result = None
        # new_vector = self.random_vector(1) * 0
        # for i in range(16):
        #     _, sample_hr = self.ema_model.sample(batch_size=1, img=new_vector)
        #     print("saving: ", i)
        #     if result is None:
        #         result = sample_hr
        #     else:
        #         result = torch.cat((result, sample_hr), dim=0)
        # utils.save_image((result + 1) /2, str(self.results_folder /  f'result.png'), nrow=4)
        # a = F.interpolate(new_vector, size=256, mode="nearest-exact")
        # utils.save_image((a + 1) / 2, str(self.results_folder / f'noise.png'), nrow=1)