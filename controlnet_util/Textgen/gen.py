import os
import cv2
import math
import numpy as np
import random
from typing import Dict

import Augmentor
import pygame
from pygame import freetype
freetype.init()
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from . import render_text_mask
from . import colorize

class datagen():
    def __init__(self, cfg: Dict, bg=None, ):
        # font init
        font_dir = cfg.font.font_dir
        self.font_list = os.listdir(font_dir)
        self.font_list = [os.path.join(font_dir, font_name) for font_name in self.font_list]
        self.font_size = cfg.font.font_size
        self.underline_rate = cfg.font.underline_rate
        self.strong_rate = cfg.font.strong_rate
        self.oblique_rate = cfg.font.oblique_rate
        # text init
        text_filepath = cfg.text.text_dir
        self.text_list = open(text_filepath, 'r').readlines()
        self.text_list = [text.strip() for text in self.text_list]
        # data augmentation
        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(probability = cfg.elastic.elastic_rate,
            grid_width = cfg.elastic.elastic_grid_size, grid_height = cfg.elastic.elastic_grid_size,
            magnitude = cfg.elastic.elastic_magnitude)
        # render rate
        self.capitalize_rate = cfg.render.capitalize_rate
        self.uppercase_rate = cfg.render.uppercase_rate
        self.curve_rate = cfg.render.curve_rate
        self.curve_rate_param = cfg.render.curve_rate_param
        self.rotate_param = cfg.render.rotate_param
        self.zoom_param = cfg.render.zoom_param
        self.shear_param = cfg.render.shear_param
        self.perspect_param = cfg.render.perspect_param
        # colorize
        self.padding_ud = cfg.colorize.padding_ud
        self.padding_lr = cfg.colorize.padding_lr
        self.is_border_rate = cfg.colorize.is_border_rate
        self.is_shadow_rate = cfg.colorize.is_shadow_rate
        self.shadow_angle_degree = cfg.colorize.shadow_angle_degree
        self.shadow_angle_param = cfg.colorize.shadow_angle_param
        self.shadow_shift_param = cfg.colorize.shadow_shift_param
        self.shadow_opacity_param = cfg.colorize.shadow_opacity_param
        # bg
        self.bg_size = (cfg.bg.width, cfg.bg.height)
        self.bg = bg
        # bbs
        self.num_bbx = cfg.bbx.num_bbx
        self.language = cfg.text.language
        
    
    def gen_text_image(self, cnt):
        font = np.random.choice(self.font_list)
        # text = self.text_list[cnt].replace('\u200c', ' ')
        text = np.random.choice(self.text_list).replace('\u200c', ' ')
        # if len(text) < 5:
        #     return False

        upper_rand = np.random.rand()
        if upper_rand < self.capitalize_rate + self.uppercase_rate:
            text = text.capitalize()
        if upper_rand < self.uppercase_rate:
            text = text.upper()

        # font and style
        font = freetype.Font(font)
        font.antialiased = True
        font.origin = True
        font.size = np.random.randint(self.font_size[0], self.font_size[1]+1)
        font.underline = np.random.rand() < self.underline_rate
        font.strong = np.random.rand() < self.strong_rate
        font.oblique = np.random.rand() < self.oblique_rate

        # render text to surf
        param = {
            'is_curve': np.random.rand() < self.curve_rate,
            'curve_rate': self.curve_rate_param[0] * np.random.rand() + self.curve_rate_param[1],
            'curve_center': np.random.randint(0, len(text))
        }
        surf, bbs = render_text_mask.render_text(font, text, param, self.language)
        
        # image = Image.fromarray(surf)
        # image.save("./results/synth/mid_wo_pers.png")
        
        # get padding
        padding_ud = np.random.randint(self.padding_ud[0], self.padding_ud[1] + 1, 2)
        padding_lr = np.random.randint(self.padding_lr[0], self.padding_lr[1] + 1, 2)
        padding = np.hstack((padding_ud, padding_lr))

        # perspect the surf
        rotate = self.rotate_param[0] * np.random.randn() + self.rotate_param[1]
        zoom = self.zoom_param[0] * np.random.randn(2) + self.zoom_param[1]
        shear = self.shear_param[0] * np.random.randn(2) + self.shear_param[1]
        perspect = self.perspect_param[0] * np.random.randn(2) +self.perspect_param[1]
        surf, bbs, canvas, bbs_8 = render_text_mask.perspective(surf, rotate, zoom, shear, perspect, padding)

        # # agument surf
        # self.surf_augmentor.augmentor_images = surf
        # surf = self.surf_augmentor.sample(1)

        # get bg size
        surf_h, surf_w = surf.shape[:2]
        surf = render_text_mask.center2size(surf, (surf_h, surf_w))

        # image = Image.fromarray(canvas)
        # image.save("./results/synth/mid_canvas.png")
        # self.visualize_poly_bb(canvas, bbs_8)

        # image = Image.fromarray(surf)
        # image.save("./results/synth/mid_surf.png")
        # self.visualize_bb(surf, bbs, 'before_color')

        bg_w, bg_h = self.bg_size
        x = np.random.randint(30, bg_w-surf_w+1-30)
        y = np.random.randint(30, bg_h-surf_h+1-30)
        surf_bg = Image.new("RGB", (surf_w,surf_h), 'white')

        # self.bg_size = (surf_h, surf_w)
        # get min h of bbs
        min_h = bbs[3]
        
        # white bg generating
        if self.bg == None:
            bg = np.array(Image.new("RGB", self.bg_size, 'white'))
        else:
            bg = np.array(self.bg)

        param = {
                        'is_border': np.random.rand() < self.is_border_rate,
                        'bordar_color': tuple(np.random.randint(100, 200, 3)),
                        'is_shadow': np.random.rand() < self.is_shadow_rate,
                        'shadow_angle': np.pi / 4 * np.random.choice(self.shadow_angle_degree)
                                        + self.shadow_angle_param[0] * np.random.randn(),
                        'shadow_shift': self.shadow_shift_param[0][:] * np.random.randn(3)
                                        + self.shadow_shift_param[1][:],
                        'shadow_opacity': self.shadow_opacity_param[0] * np.random.randn()
                                          + self.shadow_opacity_param[1]
                    }
        fg_col, bg_col = np.array((0,0,0)).astype(np.uint8), np.array((255,255,255)).astype(np.uint8)
        _, i_t = colorize.colorize(surf, surf_bg, fg_col, bg_col, min_h=min_h, param=param)

        bg[y:y+surf_h, x:x+surf_w, :] = i_t
        if self.num_bbx == 4:
            bbs = [bbs[0]+x, bbs[1]+y, bbs[2]+x, bbs[3]+y]
        elif self.num_bbx == 8:
            bbs = [bbs_8[0,0]+x, bbs_8[0,1]+y, bbs_8[1,0]+x, bbs_8[1,1]+y, bbs_8[2,0]+x, bbs_8[2,1]+y, bbs_8[3,0]+x, bbs_8[3,1]+y]
            bbs_8[:,0] = bbs_8[:,0]+x
            bbs_8[:,1] = bbs_8[:,1]+y
        # self.visualize_poly_bb(bg, bbs_8)

        return Image.fromarray(bg), [bbs], text
    
    def visualize_poly_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        cv2.polylines(ta, np.int32([bbs]), 1, color=128, thickness=2)
        plt.imsave('./results/synth/vis_bbs_8.png',ta)

    def visualize_bb(self, text_arr, bbs, name='vis_bbs_4'):
        ta = text_arr.copy()
        cv2.rectangle(ta, (bbs[0],bbs[1]), (bbs[2],bbs[3]), color=128, thickness=2)
        plt.imsave('./results/synth/'+ name +'.png',ta)
    