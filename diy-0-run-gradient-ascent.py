"""
Original CLIP Gradient Ascent Script: Used with permission by Twitter / X: @advadnoun
Heavily modified version by zer0int, https://github.com/zer0int
"""

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import kornia.augmentation as kaugs
import kornia
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style
import copy
from torch.cuda.amp import autocast, GradScaler
from safetensors.torch import load_file
scaler = GradScaler()
active_clip = None
text_pos_embed_shape = None

def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP gradient ascent')
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('--model_name_or_path', default='ViT-L/14')
    parser.add_argument('--img_folder', type=str, default="images", help="Path to image folder")
    parser.add_argument('--save_embeds', action='store_true', help="Save the embeddings to .pt files, too")
    return parser.parse_args()

os.makedirs('texts', exist_ok=True)


def infer_clip_variant(state_dict):
    # Infer the CLIP variant (e.g., ViT-L/14, ViT-B/32) based on state_dict key dimensions.
    num_layers = -1
    for key in state_dict.keys():
        if key.startswith("visual.transformer.resblocks."):
            parts = key.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                block_index = int(parts[3])
                num_layers = max(num_layers, block_index)
    
    num_layers=num_layers+1
    print(Fore.YELLOW + Style.BRIGHT + f"Model has {num_layers} layers in ViT." + Fore.RESET)

    for key, tensor in state_dict.items():
        global text_pos_embed_shape

        if not "visual." in key:
            text_pos_embed_shape = next(
                (t.size(0) for k, t in state_dict.items() if "positional_embedding" in k), 
                None
            )
        if "visual.conv1.weight" in key:
            conv_weight = tensor           
           
            # Distinguish between B/32 and B/16 using conv1 kernel size
            if conv_weight.size(2) == 32 and conv_weight.size(3) == 32:
                if text_pos_embed_shape == 248:
                    print(Fore.GREEN + Style.BRIGHT + "Identified model: Long-ViT-B/32" + Fore.RESET)
                    return "Long-ViT-B/32"
                else:
                    print(Fore.GREEN + Style.BRIGHT + "Identified model: ViT-B/32" + Fore.RESET)
                    return "ViT-B/32"
            elif conv_weight.size(2) == 16 and conv_weight.size(3) == 16:
                if text_pos_embed_shape == 248:
                    print(Fore.GREEN + Style.BRIGHT + "Identified model: Long-ViT-B/16" + Fore.RESET)
                    return "Long-ViT-B/16"
                else:
                    print(Fore.GREEN + Style.BRIGHT + "Identified model: ViT-B/16" + Fore.RESET)
                    return "ViT-B/16"
            
            # Distinguish L/14 and L/14@336 by number of layers and positional embedding
            elif conv_weight.size(2) == 14 and conv_weight.size(3) == 14:
                if num_layers == 24:
                    pos_embed_shape = next(
                        (t.size(0) for k, t in state_dict.items() if "visual.positional_embedding" in k), 
                        None
                    )
                    if pos_embed_shape == 257 and text_pos_embed_shape == 77:
                        print(Fore.GREEN + Style.BRIGHT + "Identified model: ViT-L/14" + Fore.RESET)
                        return "ViT-L/14"
                    elif pos_embed_shape == 257 and text_pos_embed_shape == 248:
                        print(Fore.GREEN + Style.BRIGHT + "Identified model: Long-ViT-L/14" + Fore.RESET)
                        return "Long-ViT-L/14"
                    elif pos_embed_shape == 577 and text_pos_embed_shape == 77:
                        print(Fore.GREEN + Style.BRIGHT + "Identified model: ViT-L/14@336px" + Fore.RESET)
                        return "ViT-L/14@336px"
                    elif pos_embed_shape == 577 and text_pos_embed_shape == 248:
                        print(Fore.GREEN + Style.BRIGHT + "Identified model: Long-ViT-L/14@336px" + Fore.RESET)
                        return "Long-ViT-L/14@336px"
                else:
                    raise ValueError(Fore.RED + Style.BRIGHT + "Unsupported CLIP-L variant." + Fore.RESET)
    raise ValueError(Fore.RED + Style.BRIGHT + "Unsupported or unknown model variant." + Fore.RESET)


def is_transformers_format(state_dict):
    # Determine if the state_dict corresponds to HuggingFace transformers format.
    return any(key.startswith("text_model") or key.startswith("vision_model") for key in state_dict.keys())

def convert_transformers_to_clip_format(state_dict):
    # Convert a HuggingFace transformers CLIP state_dict to OpenAI/CLIP format.
    clip_format = {}
    for key, value in state_dict.items():
        if key.startswith("text_model.embeddings.position_embedding.weight"):
            clip_format["positional_embedding"] = value
        elif key.startswith("vision_model.embeddings.position_embedding.weight"):
            clip_format["visual.positional_embedding"] = value
        elif key.startswith("vision_model.embeddings.patch_embedding.weight"):
            clip_format["visual.conv1.weight"] = value
        elif key.startswith("vision_model.embeddings.class_embedding"):
            clip_format["visual.class_embedding"] = value
        elif key.startswith("text_model.embeddings.token_embedding.weight"):
            clip_format["token_embedding.weight"] = value
        elif key.startswith("text_model.final_layer_norm.bias"):
            clip_format["ln_final.bias"] = value
        elif key.startswith("text_model.final_layer_norm.weight"):
            clip_format["ln_final.weight"] = value
        elif "self_attn" in key or "mlp" in key or "layer_norm" in key:
            new_key = key.replace("text_model.encoder.layers", "transformer.resblocks")
            new_key = new_key.replace("vision_model.encoder.layers", "visual.transformer.resblocks")
            new_key = new_key.replace("self_attn", "attn")
            new_key = new_key.replace("q_proj", "attn.q_proj")
            new_key = new_key.replace("k_proj", "attn.k_proj")
            new_key = new_key.replace("v_proj", "attn.v_proj")
            new_key = new_key.replace("out_proj", "attn.out_proj")
            new_key = new_key.replace("fc1", "mlp.c_fc")
            new_key = new_key.replace("fc2", "mlp.c_proj")
            new_key = new_key.replace("layer_norm1", "ln_1")
            new_key = new_key.replace("layer_norm2", "ln_2")
            clip_format[new_key] = value
        else:
            clip_format[key] = value
    return clip_format


def load_clip_model(model_name, device):
    import clip    
    global active_clip
    active_clip = clip
    """
    Load a CLIP model from a file or an available model name.
    Automatically handles .safetensors files and format conversion.
    Ensures compatibility for .pt files (full model or state_dict).
    """
    def ensure_state_dict(obj):
        """Ensure the input is a state_dict. If it's a full model object, extract the state_dict."""
        if isinstance(obj, dict):
            return obj
        elif hasattr(obj, "state_dict"):
            return obj.state_dict()
        else:
            raise ValueError(Fore.RED + Style.BRIGHT + "The provided .pt file does not contain a valid model or state_dict." + Fore.RESET)
    
    if os.path.exists(model_name):
        print(f"Loading CLIP model from path: {model_name}")
        
        if model_name.endswith(".safetensors"):
            print(f"Detected .safetensors file.")
            state_dict = load_file(model_name)
            if is_transformers_format(state_dict):
                print(Fore.CYAN + Style.BRIGHT + "Converting HuggingFace transformers model to OpenAI/CLIP format." + Fore.RESET)
                state_dict = convert_transformers_to_clip_format(state_dict)
            clip_variant = infer_clip_variant(state_dict)
            print(Fore.GREEN + Style.BRIGHT + f"Inferred CLIP variant: {clip_variant}" + Fore.RESET)

            # Handle "Long-" prefixed variants
            if clip_variant.startswith("Long-"):
                from longmodel import longclip
                active_clip = longclip
                clip_variant = clip_variant.removeprefix("Long-")  # This was just returned to sneak in Long-CLIP as active_clip, if applicable. :)
            else:
                active_clip = clip  # Use standard CLIP for non-Long variants

            try:
                model, preprocess = active_clip.load(clip_variant, device=device, jit=False)
            except TypeError:
                try: 
                    model, preprocess = active_clip.load(clip_variant, device=device)
                except FileNotFoundError:
                    try:
                        model, preprocess = active_clip.load(model_name, device=device)
                    except:
                        # Worst case scenario: Some random safetensors of a Long-CLIP with no base model. Ask user for path.
                        print(Fore.RED + Style.BRIGHT + "\n             ============== WARNING!! ==============" + Fore.RESET)
                        print(Fore.BLUE + Style.BRIGHT + "             ========= USER INPUT REQUIRED =========" + Fore.RESET)
                        last_resort_model_path = input("Sorry, this won't work without your help.\nPlease point towards the original LongCLIP model that fits the " + Fore.GREEN + Style.BRIGHT + "'Inferred CLIP Variant'" + Fore.RESET + " above.\n\nFor example, for " + Fore.GREEN + Style.BRIGHT + "Long-ViT-L/14" + Fore.RESET + ":"  + Fore.YELLOW + Style.BRIGHT + "\npath/to/LongCLIP/checkpoints/longclip-L.pt\n\n" + Fore.RESET + "...And hit Enter\n>>> ").strip()
                        if not os.path.exists(last_resort_model_path):
                            print(f"Error: The specified model path does not exist: {last_resort_model_path}")
                            return None, None
                        elif os.path.isdir(last_resort_model_path):
                            print("The specified path is a directory. Please include the model filename as well.")
                            return None, None                      
                        else:
                            model, preprocess = active_clip.load(last_resort_model_path, device=device)
                            print(Fore.MAGENTA + Style.BRIGHT + "\nYay, base model found!\nLoading custom state_dict into model..." + Fore.RESET)                           
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model = model.eval().float()
            print(Fore.MAGENTA + Style.BRIGHT + f"model.positional_embedding.shape: {model.positional_embedding.shape}")
        else:
            print(f"Assuming standard PyTorch model for loading: {model_name}")
            try:
                loaded_obj = torch.load(model_name, map_location=device, jit=False)
            except TypeError:
                loaded_obj = torch.load(model_name, map_location=device)

            state_dict = ensure_state_dict(loaded_obj)
            clip_variant = infer_clip_variant(state_dict)
            print(Fore.GREEN + Style.BRIGHT + f"Inferred CLIP variant: {clip_variant}" + Fore.RESET)

            # Handle "Long-" prefixed variants, as above in the .safetensors case
            if clip_variant.startswith("Long-"):
                from longmodel import longclip
                active_clip = longclip
                clip_variant = clip_variant.removeprefix("Long-")
                print(f"Debug: Stripped name: {clip_variant}")
            else:
                active_clip = clip  # Use standard CLIP for non-Long variants
                
            try:
                model, preprocess = active_clip.load(clip_variant, device=device, jit=False)
            except TypeError:
                try: 
                    model, preprocess = active_clip.load(clip_variant, device=device)
                except FileNotFoundError:
                    model, preprocess = active_clip.load(model_name, device=device)
                
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device).eval().float()
            print(Fore.MAGENTA + Style.BRIGHT + f"model.positional_embedding.shape: {model.positional_embedding.shape}")
    else:
        available_models = clip.available_models()
        if model_name in available_models:
            print(Fore.GREEN + Style.BRIGHT + f"Using OpenAI/CLIP model: {model_name}" + Fore.RESET)
            model, preprocess = clip.load(model_name, device)
            model = model.eval().float()
            print(Fore.MAGENTA + Style.BRIGHT + f"model.positional_embedding.shape: {model.positional_embedding.shape}")
        else:
            if not os.path.exists(model_name) and not model_name in available_models:
                raise ValueError(Fore.RED + Style.BRIGHT + f"Invalid name or path: '{model_name}'.\n\nMust be a file path [please double-check it!] or one of:\n{available_models}." + Fore.RESET)
            else:
                raise ValueError("Unknown Fatal Error")
    
    return model, preprocess


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
        
        pad_length = text_pos_embed_shape - (self.many_tokens + len(self.prompt) + 1)
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

    model_name = args.model_name_or_path                
    model, preprocess = load_clip_model(model_name, device)

    try:
        image_folder = args.img_folder
    except Exception as e:
            print(f"Please specify '--img_folder' when using '--do_batch True'\n: {e}") 
 
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                    if f.lower().endswith(valid_extensions)]
        
    for img_path in image_files:
        import clip as thetokenizer
        tok = thetokenizer.simple_tokenizer.SimpleTokenizer()
        bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None', 1005: 'None', 1006: 'None'}
        prompt = active_clip.tokenize('''''').numpy().tolist()[0]
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
