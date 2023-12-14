import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch.nn.functional as nnf
from einops import rearrange
import re
import random
# from tqdm.notebook import tqdm


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    # display(pil_img)
    pil_img.save(save_path)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    # extra_set_kwargs = {"offset": 1}
    # model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None,):
            def reshape_heads_to_batch_dim(self, tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
                return tensor

            def reshape_batch_dim_to_heads(self, tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
                return tensor
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = reshape_heads_to_batch_dim(self, q)
            k = reshape_heads_to_batch_dim(self, k)
            v = reshape_heads_to_batch_dim(self, v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            # attn = attn.softmax(dim=-1) # softmax after replace
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = reshape_batch_dim_to_heads(self, out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
            self.num_control_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet) # 改变所有的cross attention的forward函数
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

def register_controlnet_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None,):
            def reshape_heads_to_batch_dim(self, tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
                return tensor

            def reshape_batch_dim_to_heads(self, tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
                return tensor
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = reshape_heads_to_batch_dim(self, q)
            k = reshape_heads_to_batch_dim(self, k)
            v = reshape_heads_to_batch_dim(self, v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = reshape_batch_dim_to_heads(self, out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
            self.num_control_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet) # 改变所有的cross attention的forward函数
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.controlnet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count  
    

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps} # 1.
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts), max_num_words) # (51, number of prompts, 77)
    for i in range(len(prompts)):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts), 1, 1, max_num_words)
    return alpha_time_words

def aggregate_attention(attention_store, prompts, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2 # 256
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def convert_to_heatmap(image: Image.Image):
    # Convert the PIL image to grayscale
    # grayscale_image = image.convert('L')

    # # Apply a custom heatmap color map
    # heatmap = ImageOps.colorize(grayscale_image, (0, 0, 255), (255, 0, 0))

    # Convert the PIL image to a grayscale NumPy array
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply the heatmap color map
    heatmap_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_RAINBOW)

    # Convert the heatmap NumPy array back to a PIL image
    heatmap_pil = Image.fromarray(heatmap_image)

    return heatmap_pil

def show_cross_attention(attention_store, tokenizer, prompts, res: int, from_where: List[str], select: int = 0, save_path=None):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store,prompts, res, from_where, True, select) # res=16
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = convert_to_heatmap(Image.fromarray(image).resize((256, 256)))
        ############### Image to heatmap #######################
        image = np.array(image)
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    view_images(np.stack(images, axis=0),save_path=save_path)

def run_and_display(prompts, controller, pipeline, latent=None, generator=None, save_path=None):
    images, x_t = text2image_ldm_stable(pipeline, prompts, controller, latent=latent, num_inference_steps=50, guidance_scale=7.5, generator=generator)
    view_images(images, save_path=save_path)
    return images, x_t

def get_equalizer(tokenizer, text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer # change the value of the word "sign"

def get_equalizer_from_mask(tokenizer, text: str, word_select: Union[int, Tuple[int, ...]], mask: Image, device, dtype):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)

    # 1. mask image initialization
    image = [mask.convert('RGB')]
    image = [np.array(i)[None, :] for i in image]
    # image = [np.array(i.filter(ImageFilter.GaussianBlur(radius=2)))[None, :] for i in image]
    image = np.concatenate(image, axis=0)
    image = np.array(image).astype(np.float32) / 255.0
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    # image_batch_size = image.shape[0]
    # image = image.repeat_interleave(image_batch_size, dim=0)
    image = image.to(device=device, dtype=dtype)
    
    # 2. equalizer initialization
    equalizer_shape = list(image.shape)
    equalizer_shape.append(77)
    equalizer = torch.ones(equalizer_shape).to(device=device, dtype=dtype)
    select_inds = []
    for word in text.split(' '):
        inds = get_word_inds(text, word, tokenizer)
        if word in word_select:
            equalizer[:,:,:,:,inds] = image.unsqueeze(-1)
            select_inds.append(inds)
        # else:
        #     equalizer[:,:,:,:,inds] = 1.0 - image.unsqueeze(-1)
    # for word in word_select:
    #     inds = get_word_inds(text, word, tokenizer)
    #     equalizer[:,:,:,:,inds] = image.unsqueeze(-1)
    #     select_inds.append(inds)
    return equalizer, select_inds # change the value of the word "sign"

def save_tensor_as_PIL(tensor, path, name):
    import torchvision.transforms as T
    import os
    transform = T.ToPILImage()
    tensor = transform(tensor)
    tensor.save(os.path.join(path,name))

def get_self_attention_from_mask(mask, device, dtype):
    # 1. mask image initialization
    image = [mask.convert('L')]
    image = [np.array(i)[None, :] for i in image]
    image = np.concatenate(image, axis=0)
    image = np.array(image).astype(np.float32) / 255.0 # (b, h, w)
    image = torch.from_numpy(image)
    image = nnf.interpolate(image.unsqueeze(0), size=(image.shape[1]//8, image.shape[2]//8),mode='bilinear',align_corners=True).squeeze(0)
    
    image_batch_size = image.shape[0]
    image = image.to(device=device, dtype=dtype).reshape(image_batch_size, -1)

    # 2. equalizer initialization
    equalizer_shape = [*image.shape, image.shape[-1]] # (b, hw, hw)
    equalizer = torch.ones(equalizer_shape).to(device=device, dtype=dtype)
    for idx, pix in enumerate(image[0]):
        if pix > 0.5:
            equalizer[:,:,idx] = image
        else:
            equalizer[:,:,idx] = 1.0 - image
    return equalizer 

def get_draw_img_and_mask(txt_file_path, image):
    padding = 30
    img = image.copy()
    # 读取txt文件中的边界框坐标和词语
    with open(txt_file_path, 'r') as file:
        line = file.readline().strip().split(',')
        coordinates = [int(coord) for coord in line[:-1]]
        word = line[-1]

    x1_expand = random.randint(0, padding)
    x2_expand = random.randint(0, padding)
    x3_expand = random.randint(0, padding)
    x4_expand = random.randint(0, padding)
    y1_expand = random.randint(0, padding)
    y2_expand = random.randint(0, padding)
    y3_expand = random.randint(0, padding)
    y4_expand = random.randint(0, padding)

    # padding
    x1=x4=max(0, min(coordinates[::2]))  # up left x
    y1=y2=max(0, min(coordinates[1::2]))  # up left y
    x2=x3=min(img.width - 1, max(coordinates[::2]))  # down right x
    y3=y4=min(img.height - 1, max(coordinates[1::2]))  # down right y

    bbox = [x1-x1_expand, y1-y1_expand, x2+x2_expand, y2-y2_expand, \
           x3+x3_expand, y3+y3_expand, x4-x4_expand, y4+y4_expand]

    # draw and mask_draw
    draw = ImageDraw.Draw(img)
    mask_image = Image.new('1', image.size, color=0)
    mask_draw = ImageDraw.Draw(mask_image)
    polygon = []
    for i in range(0, len(bbox), 2):
        polygon.append((bbox[i], bbox[i+1]))

    # draw polygon
    draw.polygon(polygon, outline="black", width=2)
    mask_draw.polygon(polygon, fill=1)
    return img, mask_image

def generate_bounding_box_image(txt_file_path, image_size):
    padding = 30
    with open(txt_file_path, 'r') as file:
        bounding_box_coords = file.readlines()
    image = Image.new('1', image_size, color=0)
    for coord in bounding_box_coords:
        x1, y1, x2, y2 = map(int, coord.split(',')[:-1])
        for x in range(max(0,x1-padding), min(x2+padding,image_size[0]) + 1):
            for y in range(max(0,y1-(padding//2)), min(y2+(padding//2),image_size[1]) + 1):
                image.putpixel((x, y), 1) 
    return image


def generate_bounding_box_image_from_8(txt_file_path, image_size):
    padding = 30

    with open(txt_file_path, 'r') as file:
        bounding_box_coords = file.readlines()

    image = Image.new('1', image_size, color=0)
    draw = ImageDraw.Draw(image)

    for coord in bounding_box_coords:
        points = list(map(int, coord.split(',')[:-1]))
        assert len(points) == 8, "4 points is needed"

        polygon = []
        for i in range(0, len(points), 2):
            polygon.append((points[i], points[i+1]))

        draw.polygon(polygon, fill=1)

    return image

def get_bounding_box(txt_file_path, image_size):
    w, h = image_size
    with open(txt_file_path, 'r') as file:
        bounding_box_coords = file.readlines()
    coords = []
    for coord in bounding_box_coords:
        x1, y1, x2, y2 = map(int, coord.split(',')[:-1])
        coords.append([x1/w, y1/h, x2/w, y2/h])
    assert len(coords) == 1 # use only one center point
    return coords[0]

def draw_bounding_box_on_image(txt_file_path, image):
    img = image.copy()
    with open(txt_file_path, 'r') as file:
        line = file.readline().strip().split(',')
        coordinates = [int(coord) for coord in line[:-1]]
        word = line[-1]

    bbox = [
        max(0, min(coordinates[::2]) - 30),
        max(0, min(coordinates[1::2]) - 10),
        min(img.width - 1, max(coordinates[::2])),
        min(img.height - 1, max(coordinates[1::2]))
    ]

    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline='black', width=2)
    
    return img


def move_pixels(target_coords, source_coords, tensor):
    b,heads,h,w = tensor.shape
    grid = torch.zeros((b, heads, h, w, 2), dtype=tensor.dtype, device=tensor.device)
    offsets = (target_coords.to(device=tensor.device, dtype=tensor.dtype) - source_coords.to(device=tensor.device, dtype=tensor.dtype))/(h/2)
    
    grid[:, :, :, :, 0] = torch.linspace(-1, 1, h).unsqueeze(0).unsqueeze(0).unsqueeze(2)
    grid[:, :, :, :, 1] = torch.linspace(-1, 1, w).unsqueeze(0).unsqueeze(0).unsqueeze(3)
    grid += offsets.unsqueeze(2).unsqueeze(2)
    
    tensor = rearrange(tensor, "b c h w -> c b h w")
    moved_tensor = nnf.grid_sample(tensor, grid.squeeze(0), align_corners=True, padding_mode="zero")
    moved_tensor = rearrange(moved_tensor, "c b h w -> b c h w")
    
    return moved_tensor

def get_max_coordinates(tensor):
    size = tensor.shape
    # batch_size, _, height, width, _ = tensor.shape
    tensor_flat = tensor.view(size[0], size[1], size[2] * size[3])
    _, max_indices = torch.max(torch.mean(tensor_flat, dim=1, keepdim=True), dim=-1)

    row_indices = max_indices // size[-1]
    col_indices = max_indices % size[-1]

    coordinates = torch.stack((row_indices, col_indices), dim=-1)

    return coordinates

def find_words_in_string(word_list, input_string):
    found_words = []
    for word in word_list:
        if word in input_string.lower():
            found_words.append(word)
    return found_words