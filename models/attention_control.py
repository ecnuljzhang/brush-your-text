from typing import Optional, Union, Tuple, List, Callable, Dict
import abc
import torch
import torch.nn.functional as nnf
from einops import rearrange
try:
    import attention_utils
except:
    from models import attention_utils
# from models import attention_utils

MAX_NUM_WORDS = 77
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LocalBlend:
    def __init__(self, prompts: List[str], words, tokenizer, threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = attention_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t
    def between_steps(self):
        return
    @property
    def num_uncond_att_layers(self):
        return 0
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}" # cross attn (8,32*32*4,77) self attn (8, 4096,4096)
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            # self.step_store[key].append(attn.softmax(-1))
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def move_cross_attention_offset(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base = attn_repalce = attn
            # attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn = attn_repalce_new
            else:
                attn = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, tokenizer ,num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = attention_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReweight(AttentionControlEdit):

    def move_cross_attention_offset(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        # reshape attention map
        w = h = int((attn_base.size(2)) ** 0.5)
        base_shape = attn_base.shape
        # reshape cross attention to b,h,w,77
        attn_base = attn_base.reshape(*base_shape[0:2], h, w, base_shape[-1])
        x_center, y_center = (self.bbx[0]+self.bbx[2]) * w / 2, (self.bbx[1]+self.bbx[3]) * h / 2
        for idx in self.inds:
            max_idx = attention_utils.get_max_coordinates(attn_base[:,:,:,:,idx].squeeze(-1)).expand(base_shape[0], base_shape[1], -1)
            center = torch.tensor([x_center, y_center]).expand(base_shape[0], base_shape[1], -1)
            attn_base[:,:,:,:,idx] = attention_utils.move_pixels(center, max_idx, attn_base[:,:,:,:,idx].squeeze(-1)).unsqueeze(-1)
        return att_replace
        

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        # reshape attention map
        w = h = int((attn_base.size(2)) ** 0.5)
        base_shape = attn_base.shape
        attn_base = attn_base.reshape(*base_shape[0:2], h, w, base_shape[-1])
        
        # reshape equalizer
        equalizer_value = self.equalizer
        equalizer_value = rearrange(equalizer_value.unsqueeze(0), "b h w c -> b c h w")
        interpolated_tensor = nnf.interpolate(equalizer_value, size=(h,w), mode='bilinear',align_corners=True)
        interpolated_tensor = rearrange(interpolated_tensor, "b c h w -> b h w c")
        interpolated_tensor = interpolated_tensor.squeeze(0)

        # element wise multiply
        attn_replace = attn_base * interpolated_tensor[None, None,:,:,:]
        attn_replace = attn_replace.reshape(*base_shape[0:2], -1, base_shape[-1])
        for idx in self.inds:
            attn_replace[:,:,:,idx] = attn_replace[:,:,:,idx] * 6.0
            # attn_replace[:,:,:,idx] = attn_replace[:,:,:,idx] * sum(attn_base.reshape(-1,base_shape[-1])[:,idx]) / sum(attn_replace.reshape(-1,base_shape[-1])[:,idx])
        return attn_replace

    def replace_self_attention(self, attn_base, att_replace):
        if self.self_attn_equalizer is not None and attn_base.shape[2] == 64 ** 2:
            if self.prev_controller is not None:
                attn_base = self.prev_controller.replace_self_attention(attn_base, att_replace)
            base_shape = attn_base.shape # (b, 4096, 4096)
            equalizer_value = self.self_attn_equalizer
            equalizer_value = nnf.interpolate(equalizer_value.unsqueeze(0), size=base_shape[-2:], mode='bilinear',align_corners=True)
            equalizer_value = equalizer_value.squeeze(0).squeeze(0)
            attn_replace = attn_base * equalizer_value[None, None,:,:]
            return attn_replace
        else:
            return att_replace
    
    def __init__(self, prompts, tokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, select_inds,
                bbx = None, self_equalizer = None, local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.self_attn_equalizer = self_equalizer.to(device) if self_equalizer is not None else None
        self.prev_controller = controller
        self.inds = select_inds
        self.bbx = bbx