## CLIP-Interrogator-LongCLIP-hallucinwords

### CLIP Interrogator, fully in HuggingFace Transformers 🤗, with LongCLIP & CLIP's own words and / or *your* own words!

### Changes 22/JAN/25:
- `python diy-0-run-gradient-ascent.py --model_name_or_path` now loads any [OpenAI-CLIP](https://github.com/openai/CLIP) or [Long-CLIP](https://github.com/beichenzbc/Long-CLIP) model:
- Takes any .safetensors, .pt full model object OR state_dict.pt and infers which CLIP it is automatically. Except:
- Case: A random .safetensors that is a Long-CLIP. I can't know where your base models are, so I'll prompt you for that once.
------
- 🔎 Why? Because even in 2025, CLIP is still SOTA for t2i / t2v models (as a Text Encoder)!
- Refactored to use HuggingFace Transformers ✨, allows easy loading of custom CLIP models.
- Also supports Long-CLIP models with 248 tokens. And basically any other CLIP. ✨

### 😊👍
- 📄 You can insert own words with `--ownwords`. Will use `data/ownwords.txt` (put your file there!).
- You can save the results as .csv, or as .txt next to images with {image_filename}.txt. Or both.
- 📁 --output, choices=['rename', 'csv', 'txt', 'both']. For example, to save both: `--output both`
- 📉 Reduce memory use (especially for LongCLIP and / or with BLIP-2), e.g.: `--chunk_size 1024` (default 2048)
- Use `python clip-hallucin-interrogator.py --help` (or `clip-classic-interrogator.py`) for more options.


### 🤔⚠️

- `clip-classic-interrogator.py` is like the original [pharmapsychotic/clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator), just for transformers.
- `clip-hallucin-interrogator.py` uses the original + in addition, CLIP's own words, obtained in gradient ascent for many diverse images.
- ⚠️ The 'hallucin' words are NOT filtered. Will very likely contain sensitive / offensive / NSFW words that may be produced even if you have PG-13 images.
- ⚠️ Contains 'sneaky' offensive words that cannot be detected by simple NLP. Hence I cannot confidently provide a separate (safe, filtered) version.
- 🔎 A real example CLIP word: *aggravfckremove* - is: aggravated + look again: _aggrav`fck`remove_. Yes. It's a CLIP concept of 'being angry and violent'.
- ⚠️ Alas, `clip-hallucin-interrogator.py` is for research / personal (and responsible) use only.
- Remember that you can use `--mode negative` (choices=['best', 'classic', 'fast', 'negative']) to steer away from unwanted concepts with a negative prompt.
- Consider replacing `data/wCLIPy_negative.txt` with a "blacklist words github" (just google it). But still, don't use this for publicly accessible stuff.
- 👉 Also remember: Every chosen word *is* the best match for the image (according to CLIP, at least; and CLIP guides your generative model).

### ☝️🤓

```
# Example usage: 

# Fine-tuned SAE-CLIP (huggingface.co/zer0int), including additional 'trippywords':
python clip-hallucin-interrogator.py --output csv --outfile saeclipwords --image_folder images --m_clip zer0int/CLIP-SAE-ViT-L-14 --trippywords

# LongCLIP with a smaller batch size:
python clip-hallucin-interrogator.py --output csv --outfile longclipwords --image_folder images --m_clip zer0int/LongCLIP-SAE-ViT-L-14 --chunk_size 1024

# Fine-tuned GmP-CLIP with BLIP-2, save both .txt and .csv:
python clip-hallucin-interrogator.py --output both --outfile gmpblip2 --image_folder images --m_clip zer0int/CLIP-GmP-ViT-L-14 --m_caption blip2-2.7b

# Fine-tuned LongGmP-CLIP with BLIP-2 in 'fast' mode:
python clip-hallucin-interrogator.py --output csv --outfile longgmpblip2 --image_folder images --m_clip zer0int/CLIP-GmP-ViT-L-14 --m_caption blip2-2.7b --mode fast

# See all options!
python clip-hallucin-interrogator.py --help
```

### 👨‍💻🤖

- DIY wordlist using CLIP gradient ascent (yes, all above caveats apply ⚠️):
- 🔎 Why? CLIP knows best what a CLIP sees (and will subsequently guide a diffusion model into).
- But gradient ascent is expensive (compute). Give it a few representative images, get a CLIP 'opinion', re-use the words in CLIP-Interrogator for all images!✨
- Usage: `python diy-0-run-gradient-ascent.py --img_folder path/to/myimages` (or `--img_folder images` as example)
- 🆕 Optional: `--model_name_or_path` to load .safetensors or .pt of any OpenAI/CLIP or [Long-CLIP](https://github.com/beichenzbc/Long-CLIP) model
- Then: `python diy-1-preprocess-words.py`. Result: a .txt file with all the words.
- Clean them (manually review, delete weird ones), and replace `data/ownwords.txt` with the file.
- 🕵️
- Optional: If you gave CLIP a *lot* of images, you'll likely have an overwhelming amount of *words*:
- Use `python diy-2-make-clusters-DBSCAN.py` to leverage a clustering algorithm to sort them out.
- Clustering with DBSCAN will do 80% of the job, but you'll still have to categorize and review the individual clusters manually.
- [Or use the OpenAI API and ask GPT-4o or a similar SOTA model]. Ain't no simple NLP gonna understand CLIP's crazy words!
- Example for CLIP words belonging to (I guess) an 'animals' concept cluster found with DBSCAN:

![CLIP-words](https://github.com/user-attachments/assets/0e655d0c-6de4-4559-9a0b-6123b299e337)

- The script also saves plots. Check them. Very scattered clusters? Likely noise words. Is cluster AND seems to include your concept (even if in a weird way)? Valid words.
- Later clusters (higher number) are more likely to contain noise, e.g. 'asdfghj' (yes, that's a token in CLIP). Remove suspicious / unclear words.
- Edit `python diy-1-preprocess-words.py` to point to the folder with the edited clusters, save to `data/ownwords.txt`
- Remember to use the argument `--ownwords` to include your DIY words. 
- If they don't show up in the result, they were worse than the other choices in CLIP Interrogator / didn't generalize to all of your images. Try again using more images.

![example-clusters](https://github.com/user-attachments/assets/583756e1-309c-4ed9-9753-e694ed59825c)

### ✔️

```
# Change caption model, for example:
--m_caption blip2-2.7b

blip-base: 'Salesforce/blip-image-captioning-base',   # 990MB
blip-large: 'Salesforce/blip-image-captioning-large', # 1.9GB
blip2-2.7b: 'Salesforce/blip2-opt-2.7b',              # 15.5GB
blip2-flan-t5-xl 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
git-large-coco: 'microsoft/git-large-coco',           # 1.58GB
```
```
# Usage to change CLIP model (use any from HF):

--m_clip openai/clip-vit-large-patch14
--m_clip zer0int/CLIP-GmP-ViT-L-14
--m_clip zer0int/CLIP-SAE-ViT-L-14
--m_clip zer0int/LongCLIP-L-Diffusers
--m_clip zer0int/LongCLIP-GmP-ViT-L-14
--m_clip zer0int/LongCLIP-SAE-ViT-L-14
```
![demo-clip-interrogator-2025](https://github.com/user-attachments/assets/714f3fe7-ebc2-4f78-853b-c34eab7c5ce2)
