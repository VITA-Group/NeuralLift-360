from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

# import clip
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer
from torchvision import transforms

# import torch.nn.functional as F
import numpy as np

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    # print(x.shape, y.shape)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def image_similarity(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x * y).sum(-1)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, opt, device, sd_name=None):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = device
        self.opt = opt
        self.num_train_timesteps = 1000
        self.min_step = opt.min_sd
        self.max_step = opt.max_sd
        # self.min_step = int(self.num_train_timesteps * 0.02)
        # self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'[INFO] loading stable diffusion...')
        if sd_name is None:
            sd_name = 'runwayml/stable-diffusion-v1-5'
        self.sd_name = sd_name
            
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(sd_name, subfolder="vae", use_auth_token=self.token, torch_dtype=torch.float16).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_name, subfolder="tokenizer")
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_name, subfolder="text_encoder").to(self.device)
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(sd_name, subfolder="unet", use_auth_token=self.token, torch_dtype=torch.float16).to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        # for CLIP
        clip_name = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
        feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_name, torch_dtype=torch.float16)
        self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

        self.resize = transforms.Resize(224)
        self.rgb_to_latent = torch.from_numpy(np.array([[ 1.69810224, -0.28270747, -2.55163474, -0.78083445],
        [-0.02986101,  4.91430525,  2.23158593,  3.02981481],
        [-0.05746497, -3.04784101,  0.0448761 , -3.22913725]])).float().cuda(non_blocking=True) # 3 x 4
        self.latent_to_rgb = torch.from_numpy(np.array([
            [ 0.298,  0.207,  0.208],  # L1
            [ 0.187,  0.286,  0.173],  # L2
            [-0.158,  0.189,  0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
            ])).float().cuda(non_blocking=True) # 4 x 3
        # self.rgb_to_latent = self.rgb_to_latent.T # 4 x 3

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt, dir='front'):
        if True:
            # Tokenize text and get embeddings
            text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0].to(torch.float16)

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def set_epoch(self, epoch):
        self.epoch = epoch
        

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, text_z_clip=None, image_ref_clip=None, get_clip_img_embedding=None, clip_guidance_scale=100, density=None):
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        mx = self.max_step
        # mx = int(self.max_step - (self.max_step - self.min_step) / 200 * self.epoch + 0.5)
        mn = max(self.min_step, int(self.max_step - (self.max_step - self.min_step) / (self.opt.max_epoch // 3) * self.epoch + 0.5))
        # mn = self.min_step
        t = torch.randint(mn, mx + 1, [1], dtype=torch.long, device=self.device)
        use_vae = True
        if use_vae:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = pred_rgb.permute(0, 2, 3, 1) @ self.rgb_to_latent
            latents = F.interpolate(latents.permute(0, 3, 1, 2), (64, 64), mode='bilinear', align_corners=False)

        with torch.no_grad():
            alpha_prod_t = self.alphas[t]
            beta_prod_t = 1 - alpha_prod_t
            if True:
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_noisy_input = torch.cat([latents_noisy] * 2)
                latent_model_noisy_input = latent_model_noisy_input.detach().requires_grad_()
                noise_pred = self.unet(latent_model_noisy_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            ## PNDM Scheduler

        pred_original_sample = (latents_noisy - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        sample = pred_original_sample
        # sample = latents
        sample = sample.detach().requires_grad_()

        sample = 1 / 0.18215 * sample
        out_image = self.vae.decode(sample).sample
        out_image_ = (out_image / 2 + 0.5)#.clamp(0, 1)

        out_image = self.resize(out_image_)
        out_image = self.normalize(out_image)

        image_embeddings_clip = get_clip_img_embedding(out_image)
        image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
        loss_clip = 0
        if image_ref_clip is not None:
            loss_img = spherical_dist_loss(image_embeddings_clip, image_ref_clip).mean() * clip_guidance_scale * 50 # 100
            loss_clip = loss_clip + loss_img
            grads_clip = - torch.autograd.grad(loss_clip, sample, retain_graph=True)[0]
        else:
            grads_clip = 0

        with torch.no_grad():
            density = F.interpolate(density.detach(), (64, 64), mode='bilinear', align_corners=False)
            ids = torch.nonzero(density.squeeze()) 
            spatial_weight = torch.ones_like(density, device=density.device)
            try:
                up = ids[:, 0].min()
                down = ids[:, 0].max() + 1
                ll = ids[:, 1].min()
                rr = ids[:, 1].max() + 1
                spatial_weight[:, :, up:down, ll:rr] += 1
            except:
                pass

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = w * (noise_pred.detach() - noise) + w * (grads_clip.detach()) # sds loss, plus clip grad
        grad = grad * spatial_weight / 2
        
        latents.backward(gradient=grad, retain_graph=True)

        return {'clip': torch.mean(grads_clip), 'sds': (noise_pred - noise).mean(), 'sjc': noise_pred.mean()} # dummy loss value

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
