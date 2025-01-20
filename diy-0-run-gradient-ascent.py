"""
Original CLIP Gradient Ascent Script: Used with permission by Twitter / X: @advadnoun
Heavily modified version by zer0int, https://github.com/zer0int
"""

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import clip
import kornia.augmentation as kaugs
import kornia
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style
import copy
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP gradient ascent')
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('--model_name', default='ViT-L/14')
    parser.add_argument('--img_folder', type=str, default=None, help="Path to image folder")
    parser.add_argument('--save_embeds', action='store_true', help="Save the embeddings to .pt files, too")
    return parser.parse_args()

os.makedirs('texts', exist_ok=True)

def load_clip_model(model_name, device):
    model, preprocess = clip.load(model_name, device)
    return model.eval().float(), preprocess

def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std

def augment(into, augs):
    return augs(into)

def clip_encode_text(model, text, many_tokens, prompt):
    x = torch.matmul(text, model.token_embedding.weight)
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ model.text_projection
    return x

# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, bests, imagename):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]
    
    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:7], list(bests.keys())[:7]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')
        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"texts/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, batch_size, many_tokens, prompt):
        super(Pars, self).__init__()
        self.batch_size = batch_size
        self.many_tokens = many_tokens
        self.prompt = prompt
        
        # Initialize parameters
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_embeddings = torch.zeros(batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt_embeddings[:, jk, pt] = 1 
        
        self.update_padding()

    def update_padding(self):
        """Update the padding tokens based on current number of active tokens."""
        pad_length = 77 - (self.many_tokens + len(self.prompt) + 1)
        self.pad = torch.zeros(self.batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def diversity_penalty(self, new_tokens, existing_tokens, min_sim=0.6, max_sim=0.9):
        """
        Penalize new tokens for being too similar (>max_sim) or too dissimilar (<min_sim) to existing tokens.
        """
        # Compute cosine similarity between new tokens and existing tokens
        cosine_sim = F.cosine_similarity(new_tokens.unsqueeze(1), existing_tokens, dim=-1)

        # Identify where similarity is outside the acceptable range
        too_similar = (cosine_sim > max_sim).float()
        too_dissimilar = (cosine_sim < min_sim).float()

        # Penalize both cases
        penalty = too_similar * (cosine_sim - max_sim) ** 2  # Penalty for being too similar
        penalty += too_dissimilar * (min_sim - cosine_sim) ** 2  # Penalty for being too dissimilar

        # Return the mean penalty across all comparisons
        return penalty.mean()

    def add_tokens(self, num_new_tokens, model, image, optimizer, prompt, many_tokens, nom, augment):
        """Add more tokens with refined gradient-based initialization."""
        # Compute gradients for the current tokens
        loss, _, _ = ascend_txt(image, model, self, many_tokens, prompt, nom, augment)
        loss = loss.mean()  # Mean over the batch
        loss.backward()  # Compute gradients
        gradients = self.normu.grad  # Gradients w.r.t. current tokens

        # Weight gradients by their norm
        gradient_weights = gradients.norm(dim=-1, keepdim=True)  # Compute gradient magnitudes
        weighted_gradients = gradients * gradient_weights  # Scale gradients by magnitude
        weighted_mean = weighted_gradients.mean(dim=1, keepdim=True)  # Compute weighted mean

        # Use the weighted gradient mean to initialize new tokens
        new_tokens = weighted_mean.repeat(1, num_new_tokens, 1)
        new_tokens += torch.normal(mean=0, std=0.01, size=new_tokens.shape).cuda()

        # Apply diversity penalty to ensure new tokens are distinct but related
        existing_tokens = self.normu  # Existing token embeddings
        penalty = self.diversity_penalty(new_tokens, existing_tokens)
        new_tokens -= penalty * 0.1  # Adjust tokens based on penalty weight

        # Update normu with the new tokens
        self.normu = torch.nn.Parameter(torch.cat([self.normu, new_tokens], dim=1))
        self.many_tokens += num_new_tokens
        self.update_padding()

    def forward(self):
        self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat([self.start, self.prompt_embeddings, self.soft, self.pad], 1)
        return fin


# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = model.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    return -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1), tx, lll

# Loop with AMP
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll


def generate_target_text_embeddings(img_path, model, lats, optimizer, scheduler, training_iterations, checkin_step, many_tokens, prompt, nom, augment, tok, bests, args):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)

    best_loss = float('inf')  # Initialize the best loss as infinity
    best_text_embeddings = None  # Placeholder for the best text embeddings

    for j in range(training_iterations):
        # Adjust active tokens dynamically at specific steps
        if j == 50:
            num_new_tokens = 1
            print(Fore.YELLOW + Style.BRIGHT + f"Adding {num_new_tokens} tokens at step {j}..." + Fore.RESET)
            lats.add_tokens(num_new_tokens, model, img, optimizer, prompt, many_tokens, nom, augment)
                
            # Reinitialize the optimizer and scheduler with updated parameters
            optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

        # Training step
        loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
        current_loss = loss.mean().item()
            
        # Update best embeddings if current loss is better
        if current_loss < best_loss:
            best_loss = current_loss
            best_text_embeddings = copy.deepcopy(tx.detach())
            print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)
            print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)
            
        scheduler.step()

        if j % checkin_step == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)

    if args.save_embeds:
        os.makedirs('txtembeds', exist_ok=True)
        torch.save(best_text_embeddings, f"txtembeds/{img_name}_text_embedding.pt")
        print(Fore.MAGENTA + Style.BRIGHT + "\nBest text embedding saved to 'txtembeds'.\nTokens (CLIP 'opinion') saved to 'TOK'.\n" + Fore.RESET)
        
    return img, best_text_embeddings, img_path


def main():
    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"
            
    normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).cuda()
                
    model, preprocess = load_clip_model(args.model_name, device)

    try:
        image_folder = args.img_folder
    except Exception as e:
            print(f"Please specify '--img_folder' when using '--do_batch True'\n: {e}") 
 
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                    if f.lower().endswith(valid_extensions)]
        
    for img_path in image_files:
        tok = clip.simple_tokenizer.SimpleTokenizer()
        bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None', 1005: 'None', 1006: 'None'}
        prompt = clip.tokenize('''''').numpy().tolist()[0]
        prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]
                
        lats = Pars(args.batch_size, 4, prompt).cuda()
               
        augs = torch.nn.Sequential(
            kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
        ).cuda()
        
        optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)
                
        img, target_text_embedding, img_path = generate_target_text_embeddings(img_path, model, lats, optimizer, scheduler, 340, 10, 4, prompt, normalizer, augs, tok, bests, args)
        print(f"Done processing image: {img_path}; CLIP opinion saved to the 'texts' folder.")

   
if __name__ == "__main__":
    main()
