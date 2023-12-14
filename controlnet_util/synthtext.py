from PIL import Image, ImageDraw, ImageFont
import numpy as np
from Textgen.gen import datagen
from omegaconf import OmegaConf
from typing import Dict
import os

def text_image(text, cor_xy, image_size, font_path='arial.ttf', font_size=16, background_color='white', text_color='black'):
    left, top, right, bottom = cor_xy
    width = right - left
    height = bottom - top
    
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    font_size = font_size * min(width / text_width, height / text_height)
    font = ImageFont.truetype(font_path, int(font_size))

    draw.text((left, top), text, fill=text_color, font=font)
    return image
    
def main(cfg: Dict):
    data_gen = datagen(cfg)
    cnt = 0
    result_path = cfg.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # if cfg.read_from_prompt
    #     assert cfg.prompt_path is not None
    while(cnt < cfg.sample_num):
        try:
            text_image, bbs, text = data_gen.gen_text_image(cnt)
        except:
            continue
        else:
            image_name = f"{cnt}.png"
            text_name = f"{cnt}.txt"
            text_image.save(os.path.join(result_path, image_name))
            with open(os.path.join(result_path, text_name), "w") as f:
                for bb in bbs:
                    line = ",".join(str(x) for x in bb)
                    f.write(line + ',' + text + '\n')
            cnt += 1
    

if __name__ == '__main__':
    main(**OmegaConf.load('/path/to/controlnet_util/Textgen/text_cfg.yaml'))
    print('generation finished.')