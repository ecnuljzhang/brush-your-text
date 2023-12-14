import os
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Union

from tqdm import tqdm
import numpy as np

#################### images saving ######################
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image, path, name):
    make_dir(path)
    image.save(os.path.join(path, name))

def save_grid(imgs, rows, cols, names, path, cnt):
    assert len(imgs) == rows * cols
    assert len(names) == rows
    for i, img in enumerate(imgs):
        save_image(img, path, cnt + '-' + names[i // cols] + '-' + str(i % cols) + '.png')
    return
###########################################################

#################### DDIM Inversion #######################
# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt)
    return ddim_latents
###########################################################

def get_mask_and_text_image(image_path, bbx_path, word_list, font_type):
    # Step 1: Read bounding box data from the bbx_path
    with open(bbx_path, 'r') as bbx_file:
        bbx_data = bbx_file.readlines()

    # Step 2: Read the image and get its width and height
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Step 3: Create the mask image
    mask = Image.new('1', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    for bbx_line in bbx_data:
        bbx_info = bbx_line.strip().split(' ')
        if len(bbx_info) == 5:
            x_min, y_min, x_max, y_max = map(int, bbx_info[:-1])
            draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=1)

    # Step 4: Create the text image with words in the word_list
    text_image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    text_draw = ImageDraw.Draw(text_image)

    for idx, bbx_line in enumerate(bbx_data):
        bbx_info = bbx_line.strip().split(' ')
        if len(bbx_info) == 5:
            x_min, y_min, x_max, y_max, word = bbx_info
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            word_to_draw = word_list[idx % len(word_list)]

            # Adjust font size to fit within the bounding box
            font_size = 1
            font = ImageFont.truetype(font_type, font_size)
            while text_draw.textsize(word_to_draw, font=font)[0] < (x_max - x_min) and \
                    text_draw.textsize(word_to_draw, font=font)[1] < (y_max - y_min):
                font_size += 1
                font = ImageFont.truetype(font_type, font_size)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_draw = x_center - text_draw.textsize(word_to_draw, font=font)[0] / 2
            y_draw = y_center - text_draw.textsize(word_to_draw, font=font)[1] / 2

            text_draw.text((x_draw, y_draw), word_to_draw, font=font, fill=(0, 0, 0))

    return mask, text_image