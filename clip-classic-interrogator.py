"""
Original CLIP Interrogator file:
    https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb

# CLIP Interrogator 2.4 by [@pharmapsychotic](https://twitter.com/pharmapsychotic)
https://github.com/pharmapsychotic/clip-interrogator

Refactored to use HuggingFace transformers instead of open_clip by zer0int
https://github.com/zer0int
"""

import os
import subprocess
import contextlib
import io
import sys
import re
from colorama import Fore, Style
from functools import wraps
import argparse
import csv
import requests
import time
import hashlib
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Optional
from dataclasses import dataclass
from safetensors.numpy import load_file, save_file
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from transformers import CLIPVisionModel, CLIPVisionConfig
from transformers.utils import logging
logging.set_verbosity_error()

"""
                            ------------- TO-DO --------------

- Check flaves correctly working with normal interrogate (not just fast)

- Make quiet so that it still prints the prompt AND the current image

- Ensure custom HF can be loaded (for CLIP; check caption models?)

- Check num_features and how that may be used?!

"""
def validate_clip_model(model_name):
    predefined_models = [
        'openai/clip-vit-large-patch14',# Original CLIP
        'zer0int/CLIP-GmP-ViT-L-14',# My GmP finetune
        'zer0int/CLIP-SAE-ViT-L-14',# My SAE finetune
        'zer0int/LongCLIP-L-Diffusers',# Original LongCLIP
        'zer0int/LongCLIP-GmP-ViT-L-14',# My GmP-LongCLIP
        'zer0int/LongCLIP-SAE-ViT-L-14'# My SAE-LongCLIP 
        # Just use this argument to load a different CLIP from HuggingFace: --m_clip SomeDev/Some-CLIP-model
    ]

    if model_name in predefined_models:
        return model_name
    else:
        # Attempt to check if the user-defined model is accessible
        try:
            AutoConfig.from_pretrained(model_name)
            return model_name  # Valid custom model
        except Exception as e:
            if "404" in str(e):
                print(f"Error: Model '{model_name}' not found.")
            elif "gated" in str(e).lower():
                print(f"Error: Model '{model_name}' is gated and requires access permissions.")
            else:
                print(f"{Fore.RED + Style.BRIGHT}Error: Unable to load model '{model_name}' due to: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW + Style.BRIGHT}Falling back to default model: 'openai/clip-vit-large-patch14'{Style.RESET_ALL}")
            return 'openai/clip-vit-large-patch14'

def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP interrogator 2025')
    parser.add_argument('--m_caption', choices=['blip-base', 'blip-large', 'blip2-2.7b', 'blip2-flan-t5-xl', 'git-large-coco'], default='blip-large', type=str, help="Caption Model to use")
    parser.add_argument('--m_clip', type=validate_clip_model, default='openai/clip-vit-large-patch14', help="Specify the HuggingFace CLIP model to use. For example: --m_clip zer0int/CLIP-GmP-ViT-L-14")
    parser.add_argument('--mode', choices=['best', 'classic', 'fast', 'negative'], default='best', type=str, help="Mode to use for Captioning")
    parser.add_argument('--ownwords', action='store_true', help="Use ownwords.txt (put your own file in 'data/ownwords.txt' first!)")
    parser.add_argument('--output', choices=['rename', 'csv', 'txt', 'both'], default='csv', type=str, help="Rename image filenames, save captions as .csv, as individual .txt files, or as both .csv + .txt")
    parser.add_argument('--outfile', type=str, default='all', help="Filename for .csv; defaults to 'all' -> 'all.csv'")
    parser.add_argument('--maxfilename', default=48, type=int, help="Maximum caption / filename length (default: 48), only applies if used with: --output rename")
    parser.add_argument('--maxcaption', default=32, type=int, help="Maximum BLIP caption length, default: 32")
    parser.add_argument('--chunk_size', default=2048, type=int, help="Batch size, default: 2048; reduce to e.g. 1024 or 512 to use less VRAM")
    parser.add_argument('--max_flavors', default=64, type=int, help="Maximum flavors in the Flavor Chain; default: 64")
    parser.add_argument('--image_folder', type=str, default='images', help="Defaults to 'images'; expects: /path/to/image/folder")
    parser.add_argument('--quiet', action='store_true', help="Run quietly, without verbose output")
    parser.add_argument('--ranklooker', action='store_true', help="Watch the ranking process, live! Can slow down the process a bit; only use if you're actually looking. =)")
    parser.add_argument('--debug', action='store_true', help="Spams everything with tensors & current stats. Not useful - except for DEBUGGING.")
    return parser.parse_args()

args = parse_arguments()

maxcaption = args.maxcaption
caption_model_name = args.m_caption
clip_model_name = args.m_clip
max_flavors=args.max_flavors

default_max_tokens = 77
max_tokens = default_max_tokens
dumptokensconfig = AutoConfig.from_pretrained(clip_model_name)
if hasattr(dumptokensconfig, "text_config") and hasattr(dumptokensconfig.text_config, "max_position_embeddings"):
    max_tokens = dumptokensconfig.text_config.max_position_embeddings
else:
    max_tokens = default_max_tokens
del dumptokensconfig

print(f"{Fore.MAGENTA + Style.BRIGHT}\nSet {clip_model_name} with max_tokens: {max_tokens}{Style.RESET_ALL}")

CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
    'git-large-coco': 'microsoft/git-large-coco',           # 1.58GB
}

@dataclass 
class Config:
    # models can optionally be passed in directly
    caption_model = None
    caption_processor = None
    clip_model = None
    clip_preprocess = None

    # blip settings
    caption_max_length: int = maxcaption
    caption_model_name: Optional[str] = caption_model_name
    caption_offload: bool = False

    # clip settings
    clip_model_name: str = clip_model_name
    clip_model_path: Optional[str] = None
    clip_offload: bool = False

    # interrogator settings
    cache_path: str = 'cache'   # path to store cached text embeddings
    chunk_size: int = args.chunk_size      # batch size for CLIP, use smaller for lower VRAM
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    flavor_intermediate_count: int = 2048
    quiet: bool = args.quiet # when quiet, progress bars are not shown


def create_tokenizer(clip_model_name: str, max_length: int = max_tokens):
    """
    Creates a tokenizer function that replicates open_clip's behavior using the Hugging Face CLIPTokenizer.
    """
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    tokenizer.pad_token = "0" # Ensure padding token is set to 0 (instead of eos_token)

    def tokenize_text(text_array):
        if isinstance(text_array, str):
            text_array = [text_array] # Convert single string to a list, if needed
        # Tokenize and pad/truncate
        tokenized = tokenizer(
            text_array,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # Ensure that the padding token is explicitly set to 0
        input_ids = tokenized["input_ids"]
        input_ids[input_ids == tokenizer.pad_token_id] = 0
        
        # Convert to padded tensor format matching open_clip
        padded_tensor = torch.zeros((input_ids.size(0), max_length), dtype=torch.long)
        padded_tensor[:, :input_ids.size(1)] = input_ids
        return padded_tensor

    return tokenize_text


class Interrogator():
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.caption_offloaded = True
        self.clip_offloaded = True
        self.load_caption_model()
        self.load_clip_model()

    def load_caption_model(self):
        if self.config.caption_model is None and self.config.caption_model_name:
            if not self.config.quiet:
                print(f"{Fore.GREEN + Style.BRIGHT}Loading caption model {self.config.caption_model_name}...{Style.RESET_ALL}")

            model_path = CAPTION_MODELS[self.config.caption_model_name]
            if self.config.caption_model_name.startswith('git-'):
                caption_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
            elif self.config.caption_model_name.startswith('blip2-'):
                caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
            else:
                caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
            self.caption_processor = AutoProcessor.from_pretrained(model_path)

            caption_model.eval()
            if not self.config.caption_offload:
                caption_model = caption_model.to(self.config.device)
            self.caption_model = caption_model
        else:
            self.caption_model = self.config.caption_model
            self.caption_processor = self.config.caption_processor

    def load_clip_model(self):
        start_time = time.time()
        config = self.config

        if config.clip_model is None:
            if not config.quiet:
                print(f"{Fore.GREEN + Style.BRIGHT}Loading CLIP model {config.clip_model_name}...{Style.RESET_ALL}")

            processor = CLIPProcessor.from_pretrained(config.clip_model_name)
            clip_model = CLIPModel.from_pretrained(config.clip_model_name)

            clip_model.eval()
            if not config.clip_offload:
                clip_model = clip_model.to(config.device)

            self.clip_model = clip_model
            self.clip_processor = processor          
            
            self.clip_model.eval()
        else:
            self.clip_model = config.clip_model
            self.clip_preprocess = config.clip_preprocess
        self.tokenize = create_tokenizer(clip_model_name)      
     

        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 
                 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 
                 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on "+site for site in sites])
        trending_list.extend(["featured on "+site for site in sites])
        trending_list.extend([site+" contest winner" for site in sites])

        raw_artists = load_list(config.data_path, 'artists.txt')
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        self._prepare_clip()
        self.artists = LabelTable(artists, "artists", self)
        self.flavors = LabelTable(load_list(config.data_path, 'flavors.txt'), "flavors", self)
        self.mediums = LabelTable(load_list(config.data_path, 'mediums.txt'), "mediums", self)
        self.movements = LabelTable(load_list(config.data_path, 'movements.txt'), "movements", self)
        self.trendings = LabelTable(trending_list, "trendings", self)
        self.negative = LabelTable(load_list(config.data_path, 'negative.txt'), "negative", self)
        if args.ownwords:
            self.ownwords = LabelTable(load_list(config.data_path, 'ownwords.txt'), "ownwords", self)
            
        end_time = time.time()
        if not config.quiet:
            print(f"{Fore.MAGENTA + Style.BRIGHT}Loaded CLIP model and data in {end_time-start_time:.2f} seconds.{Style.RESET_ALL}\n")

    def chain(
        self, 
        image_features: torch.Tensor, 
        phrases: List[str], 
        best_prompt: str="", 
        best_sim: float=0, 
        min_count: int=8,
        max_count: int=64, 
        desc="Chaining", 
        reverse: bool=False
    ) -> str:
        self._prepare_clip()

        phrases = set(phrases)
        if not best_prompt:
            best_prompt = self.rank_top(image_features, [f for f in phrases], reverse=reverse)
            best_sim = self.similarity(image_features, best_prompt)
            phrases.remove(best_prompt)
        curr_prompt, curr_sim = best_prompt, best_sim
        
        def check(addition: str, idx: int) -> bool:
            nonlocal best_prompt, best_sim, curr_prompt, curr_sim
            prompt = curr_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if reverse:
                sim = -sim
            
            if sim > best_sim:
                best_prompt, best_sim = prompt, sim
            if sim > curr_sim or idx < min_count:
                curr_prompt, curr_sim = prompt, sim
                return True
            return False

        if args.ranklooker:
            pbar = tqdm(range(max_count), desc=desc, disable=self.config.quiet)
            for idx in pbar:
                processed_phrases = [f"{curr_prompt}, {f} " for f in phrases]
                pbar.set_description(f"{Fore.MAGENTA + Style.BRIGHT}Ranklooker: {Fore.CYAN + Style.BRIGHT}{processed_phrases[:5]}{Style.RESET_ALL}")  # Update the description dynamically
                
                best = self.rank_top(image_features, processed_phrases, reverse=reverse)              
                flave = best[len(curr_prompt)+2:]
                flavealt = best[len(curr_prompt)+2:-1] 
                if not check(flave, idx):
                    break
                if _prompt_at_max_len(curr_prompt, self.tokenize):
                    break
                try:
                    phrases.remove(flave)
                except KeyError:
                    flave = flavealt 
                    phrases.remove(flave)
            pbar.close()  # Explicitly close the progress bar
            return best_prompt

        else:
            for idx in tqdm(range(max_count), desc=desc, disable=self.config.quiet):
                
                best = self.rank_top(image_features, [f"{curr_prompt}, {f}" for f in phrases], reverse=reverse)
                flave = best[len(curr_prompt)+2:]
                flavealt = best[len(curr_prompt)+2:-1] 
                if not check(flave, idx):
                    break
                if _prompt_at_max_len(curr_prompt, self.tokenize):
                    break
                try:
                    phrases.remove(flave)
                except KeyError:
                    flave = flavealt 
                    phrases.remove(flave)
            return best_prompt

    def generate_caption(self, pil_image: Image) -> str:
        assert self.caption_model is not None, "No caption model loaded."
        self._prepare_caption()
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(self.device)
        if not self.config.caption_model_name.startswith('git-'):
            inputs = inputs.to(self.dtype)
        tokens = self.caption_model.generate(**inputs, max_new_tokens=self.config.caption_max_length)
        return self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def image_to_features(self, image: Image) -> torch.Tensor:
        self._prepare_clip()
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        images = inputs["pixel_values"].to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = self.clip_model.get_image_features(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def interrogate_classic(self, image: Image, max_flavors: int=max_flavors, caption: Optional[str]=None) -> str:
        """Classic mode creates a prompt in a standard format first describing the image, 
        then listing the artist, trending, movement, and flavor text modifiers."""
        caption = caption or self.generate_caption(image)
        image_features = self.image_to_features(image)        

        medium = self.mediums.rank(image_features, 1)[0]
        artist = self.artists.rank(image_features, 1)[0]
        trending = self.trendings.rank(image_features, 1)[0]
        movement = self.movements.rank(image_features, 1)[0]
        flaves = ", ".join(self.flavors.rank(image_features, max_flavors))
        if args.ownwords:
            ownwords = ", ".join(self.ownwords.rank(image_features, max_flavors))
        if args.debug:
            print(f"def interrogate_classic, Ranked labels: {len(flaves)}, Top-ranked: {flaves[:5]}")

        if caption.startswith(medium):
            prompt = f"{caption} {artist}, {trending}, {movement}, {flaves}"
        elif caption.startswith(medium) and args.ownwords:
            prompt = f"{caption} {artist}, {ownwords}, {trending}, {movement}, {flaves}"
        elif args.ownwords:
            prompt = f"{caption}, {ownwords}, {medium} {artist}, {trending}, {movement}, {flaves}"
        else:
            prompt = f"{caption}, {medium} {artist}, {trending}, {movement}, {flaves}"
            
        return _truncate_to_fit(prompt, self.tokenize)

    def interrogate_fast(self, image: Image, max_flavors: int=max_flavors, caption: Optional[str]=None) -> str:
        """Fast mode simply adds the top ranked terms after a caption. It generally results in 
        better similarity between generated prompt and image than classic mode, but the prompts
        are less readable."""
        caption = caption or self.generate_caption(image)
        image_features = self.image_to_features(image)
        
        if args.ownwords:
            merged = _merge_tables([self.artists, self.ownwords, self.flavors, self.mediums, self.movements, self.trendings], self)
        else:
            merged = _merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self)
        
        tops = merged.rank(image_features, max_flavors)
        return _truncate_to_fit(caption + ", " + ", ".join(tops), self.tokenize)

    def interrogate_negative(self, image: Image, max_flavors: int = 32) -> str:
        """Negative mode chains together the most dissimilar terms to the image. It can be used
        to help build a negative prompt to pair with the regular positive prompt and often 
        improve the results of generated images."""
        image_features = self.image_to_features(image)
        flaves = self.flavors.rank(image_features, self.config.flavor_intermediate_count, reverse=True)
        flaves = flaves + self.negative.labels
        return self.chain(image_features, flaves, max_count=max_flavors, reverse=True, desc="Negative chain")

    def interrogate(self, image: Image, min_flavors: int=8, max_flavors: int=max_flavors, caption: Optional[str]=None) -> str:
        caption = caption or self.generate_caption(image)
        image_features = self.image_to_features(image)

        if args.ownwords:
            merged = _merge_tables([self.artists, self.ownwords, self.flavors, self.mediums, self.movements, self.trendings], self)
        else:
            merged = _merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self)
            
        if args.debug:
            print(f"def interrogate, Merged labels: {len(merged.labels)}")
            
        flaves = merged.rank(image_features, self.config.flavor_intermediate_count)
        if not self.config.quiet:
            print(f"{Fore.CYAN + Style.BRIGHT}Flaves (total): {Fore.YELLOW + Style.BRIGHT}{len(flaves)}. {Fore.CYAN + Style.BRIGHT}Tops: {Fore.YELLOW + Style.BRIGHT}{flaves[:8]}{Style.RESET_ALL}")
        best_prompt, best_sim = caption, self.similarity(image_features, caption)
        best_prompt = self.chain(image_features, flaves, best_prompt, best_sim, min_count=min_flavors, max_count=max_flavors, desc="Flavor chain")

        fast_prompt = self.interrogate_fast(image, max_flavors, caption=caption)
        classic_prompt = self.interrogate_classic(image, max_flavors, caption=caption)
        candidates = [caption, classic_prompt, fast_prompt, best_prompt]
        
        return candidates[np.argmax(self.similarities(image_features, candidates))]

    def rank_top(self, image_features: torch.Tensor, text_array: List[str], reverse: bool=False) -> str:
        self._prepare_clip()
        text_tokens = self.tokenize([text for text in text_array]).to(self.device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.clip_model.get_text_features(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T

            if reverse:
                similarity = -similarity
        
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        self._prepare_clip()
        text_tokens = self.tokenize([text]).to(self.device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):

            text_features = self.clip_model.get_text_features(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        
        return similarity[0][0].item()

    def similarities(self, image_features: torch.Tensor, text_array: List[str]) -> List[float]:
        self._prepare_clip()
        text_tokens = self.tokenize([text for text in text_array]).to(self.device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.clip_model.get_text_features(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        
        return similarity.T[0].tolist()

    def _prepare_caption(self):
        if self.config.clip_offload and not self.clip_offloaded:
            self.clip_model = self.clip_model.to('cpu')
            self.clip_offloaded = True
        if self.caption_offloaded:
            self.caption_model = self.caption_model.to(self.device)
            self.caption_offloaded = False

    def _prepare_clip(self):
        if self.config.caption_offload and not self.caption_offloaded:
            self.caption_model = self.caption_model.to('cpu')
            self.caption_offloaded = True
        if self.clip_offloaded:
            self.clip_model = self.clip_model.to(self.device)
            self.clip_offloaded = False


class LabelTable():
    def __init__(self, labels:List[str], desc:str, ci: Interrogator):
        clip_model, config = ci.clip_model, ci.config
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.labels = labels
        self.tokenize = ci.tokenize
        self.clip_processor = ci.clip_processor

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()
        sanitized_name = self.config.clip_model_name.replace('/', '_').replace('@', '_')
        self._load_cached(desc, hash, sanitized_name)

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels) / config.chunk_size))

            if args.debug:
                print(f"Total labels: {len(self.labels)}, Chunk size: {self.chunk_size}, Num chunks: {len(chunks)}")
                for i, chunk in enumerate(chunks):
                    print(f"Chunk {i}: {len(chunk)} items")

            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                chunk = list(chunk)
                text_tokens = self.tokenize(chunk).to(self.device)

                with torch.no_grad(), torch.amp.autocast('cuda'):
                    
                    text_features = clip_model.get_text_features(text_tokens)
                    if args.debug:
                        print(f"LabelTable Embedding norms: {np.linalg.norm(text_features, axis=-1)}")
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    if args.debug:
                        print(f"LabelTable Embedding norms after .norm: {np.linalg.norm(text_features, axis=-1)}")
                    text_features = text_features.half().cpu().numpy()

                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if desc and self.config.cache_path:
                os.makedirs(self.config.cache_path, exist_ok=True)
                cache_filepath = os.path.join(self.config.cache_path, f"{sanitized_name}_{desc}.safetensors")
                tensors = {
                    "embeds": np.stack(self.embeds),
                    "hash": np.array([ord(c) for c in hash], dtype=np.int8)
                }
                save_file(tensors, cache_filepath)
                torch.cuda.empty_cache()

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.embeds = [e.astype(np.float32) for e in self.embeds]

    def _load_cached(self, desc:str, hash:str, sanitized_name:str) -> bool:
        if self.config.cache_path is None or desc is None:
            return False

        cached_safetensors = os.path.join(self.config.cache_path, f"{sanitized_name}_{desc}.safetensors")        

        if os.path.exists(cached_safetensors):
            try:
                tensors = load_file(cached_safetensors)
            except Exception as e:
                print(f"Failed to load {cached_safetensors}")
                print(e)
                return False
            if 'hash' in tensors and 'embeds' in tensors:
                if np.array_equal(tensors['hash'], np.array([ord(c) for c in hash], dtype=np.int8)):
                    self.embeds = tensors['embeds']
                    if len(self.embeds.shape) == 2:
                        self.embeds = [self.embeds[i] for i in range(self.embeds.shape[0])]
                    return True

        return False
    
    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1, reverse: bool=False) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        
        if args.debug:
            print(f"def _rank: Image features norm: {torch.norm(image_features, dim=-1).cpu().numpy()}")
            print(f"def _rank: Text embeddings norm (batch): {torch.norm(text_embeds, dim=-1).cpu().numpy()}")

        with torch.amp.autocast('cuda'):
            similarity = image_features @ text_embeds.T
            if args.debug:
                print(f"Similarity scores (batch): {similarity.cpu().numpy()}")            
            
            if reverse:
                similarity = -similarity
                
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int=5, reverse: bool=False) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count, reverse=reverse)
            if args.debug:
                print(f"def rank: Top labels: {tops}")
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        if args.debug:
            print(f"Splitting into {num_chunks} chunks...")
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeds))
            chunk_embeds = self.embeds[start:stop]
            if args.debug:
                print(f"Processing chunk {chunk_idx}: {len(chunk_embeds)} embeddings")
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk, reverse=reverse)
            if args.debug:
                print(f"Chunk {chunk_idx} top labels count: {len(tops)}")
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        if args.debug:
            print(f"Final top labels count: {len(tops)}")
        return [top_labels[i] for i in tops]

def _merge_tables(tables: List[LabelTable], ci: Interrogator) -> LabelTable:
    m = LabelTable([], None, ci)
    
    for table in tables:
        if args.debug:
            print(f"Merging {len(table.labels)} labels from table: {table}")
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    if args.debug:
        print(f"Total merged labels: {len(m.labels)}, Total embeddings: {len(m.embeds)}")
    return m

def _prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0
    
def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        candidate = new_text + ', ' + part
        if _prompt_at_max_len(candidate, tokenize):
            break
        new_text = candidate
    return new_text

def load_list(data_path: str, filename: Optional[str] = None) -> List[str]:
    if filename is not None:
        data_path = os.path.join(data_path, filename)
    with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def image_to_prompt(image, imgname, mode):
    ci.config.chunk_size = args.chunk_size
    ci.config.flavor_intermediate_count = 2048
    image = image.convert('RGB')
    if mode == 'best':
        print(f"{Fore.YELLOW + Style.BRIGHT}\n\nProcessing image: {Fore.MAGENTA + Style.BRIGHT}{imgname}{Style.RESET_ALL}")
        return ci.interrogate(image)
    elif mode == 'classic':
        print(f"{Fore.YELLOW + Style.BRIGHT}\n\nProcessing image: {Fore.MAGENTA + Style.BRIGHT}{imgname}{Style.RESET_ALL}")
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        print(f"{Fore.YELLOW + Style.BRIGHT}\n\nProcessing image: {Fore.MAGENTA + Style.BRIGHT}{imgname}{Style.RESET_ALL}")
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        print(f"{Fore.YELLOW + Style.BRIGHT}\n\nProcessing image: {Fore.MAGENTA + Style.BRIGHT}{imgname}{Style.RESET_ALL}")
        return ci.interrogate_negative(image)

def sanitize_for_filename(prompt: str, max_len: int) -> str:
    name = "".join(c for c in prompt if (c.isalnum() or c in ",._-! "))
    name = name.strip()[:(max_len-4)] # space for extension
    return name

def get_csv_filename(outfile, default="all.csv"):
    if not outfile:
        return default
    if not outfile.endswith(".csv"):
        outfile = os.path.splitext(outfile)[0] + ".csv"
    return outfile


config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

folder_path = args.image_folder
prompt_mode = args.mode
output_mode = args.output
max_filename_len = args.maxfilename

files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(folder_path) else []
prompts = []

for idx, file in enumerate(tqdm(files, desc='Generating prompts')):
    image = Image.open(os.path.join(folder_path, file)).convert('RGB')
    imgname = os.path.basename(os.path.splitext(file)[0])
    prompt = image_to_prompt(image, imgname, prompt_mode)
    prompts.append(prompt)

    print(f"{Fore.GREEN + Style.BRIGHT}{prompt}{Style.RESET_ALL}")

    if output_mode == 'rename':
        name = sanitize_for_filename(prompt, max_filename_len)
        ext = os.path.splitext(file)[1]
        filename = name + ext
        idx = 1
        while os.path.exists(os.path.join(folder_path, filename)):
            print(f'{Fore.YELLOW + Style.BRIGHT}File {filename} already exists, trying {idx+1}...{Style.RESET_ALL}"')
            filename = f"{name}_{idx}{ext}"
            idx += 1
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, filename))

if len(prompts):
    if output_mode == 'csv':
        csvfile = get_csv_filename(args.outfile)
        csv_path = os.path.join(folder_path, csvfile)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(['image', 'prompt'])
            for file, prompt in zip(files, prompts):
                cleaned_prompt = re.sub(r'\s+,', ',', prompt)
                w.writerow([file, cleaned_prompt])

        print(f"{Fore.GREEN + Style.BRIGHT}\n\n\n\nGenerated {len(prompts)} prompts and saved to '{csv_path}', enjoy!{Style.RESET_ALL}")
    
    elif output_mode == 'txt':
        for file, prompt in zip(files, prompts):
            txt_filename = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(folder_path, txt_filename)
            formatted_prompt = "\n".join([line.strip() for line in prompt.split(",")])
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(formatted_prompt)

        print(f"{Fore.GREEN + Style.BRIGHT}\n\n\n\nGenerated {len(prompts)} prompts and saved to '{folder_path}' as .txt files, enjoy!{Style.RESET_ALL}")

    elif output_mode == 'both':
        for file, prompt in zip(files, prompts):
            txt_filename = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(folder_path, txt_filename)
            formatted_prompt = "\n".join([line.strip() for line in prompt.split(",")])
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(formatted_prompt)        

        csvfile = get_csv_filename(args.outfile)
        csv_path = os.path.join(folder_path, csvfile)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(['image', 'prompt'])
            for file, prompt in zip(files, prompts):
                cleaned_prompt = re.sub(r'\s+,', ',', prompt)
                w.writerow([file, cleaned_prompt])

        print(f"{Fore.GREEN + Style.BRIGHT}\n\n\n\nGenerated {len(prompts)} prompts and saved to '{folder_path}' as .txt files + saved to '{csv_path}', enjoy!{Style.RESET_ALL}")
    
    else:
        print(f"{Fore.GREEN + Style.BRIGHT}\n\n\n\nGenerated {len(prompts)} prompts and renamed your files, enjoy!{Style.RESET_ALL}")
else:
    print(f"{Fore.RED + Style.BRIGHT}No images in {folder_path}!{Style.RESET_ALL}")
