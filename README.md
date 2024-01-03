# Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model (AAAI2024)
Lingjun Zhang*, [Xinyuan Chen*](https://scholar.google.com/citations?user=3fWSC8YAAAAJ&hl=zh-CN), [Yaohui Wang](https://wyhsirius.github.io/), [Yue Lu](https://scholar.google.com/citations?user=_A_H0V4AAAAJ&hl=zh-CN) and [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en) (* indicates equal contribution)

This is the official PyTorch implementation of the AAAI 2024 paper ["Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model"](https://arxiv.org/abs/2312.12232). <br>

<img src="/pics/teaser.png" width="800px">

# Getting Start
## Installation
```
git clone https://github.com/ecnuljzhang/brush-your-text.git
cd brush-your-text
```
## Environment
```
conda create -n difftext python=3.8.5
conda activate difftext
pip install -r requirements.txt
```

## Checkpoints
The pre-trained checkpoints can be downloaded from [Hugging Face](https://huggingface.co/). Please download ["runwayml/stable-diffusion-v1-5"](https://huggingface.co/runwayml/stable-diffusion-v1-5) and ["lllyasviel/sd-controlnet-canny"](https://huggingface.co/lllyasviel/sd-controlnet-canny) models and put them into the "checkpoints" folder. The file structures could be as follows:
```
Diff-text
|--- checkpoints
|          |--- stable-diffusion-v1-5/
|          |--- sd-controlnet-canny/
|--- ...
```

# How to Run
## 1. Prepare Sketch Images
You may prepare the sketch image yourself (as shown in the first row of the image below) along with the corresponding bounding box text file (please use the ICDAR2013 or ICDAR2015 dataset annotation format), or utilize our sketch image synthesis code. The synthesis code can be executed with the following command:
```
python controlnet_util/synthtext.py
```
Before running the synthesis code, please make the necessary modifications to the "controlnet_util/Textgen/text_cfg.yaml" file, following the instructions provided in the file comments. <br>
<img src="/pics/sketch_img.png" width="800px">
## 2. Prepare Prompts
You can create a text file containing prompts, with the file extension ".txt".
## 3. Modify Configurations
Remember to make the necessary modifications to the "configs/control_gen.yaml", following the instructions provided in the file comments.
## 4. Run
```
python predict.py
```

# Citation
If you find this code useful in your research, please consider citing:
```bibtex
@article{zhang2023brush,
      title={Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model}, 
      author={Lingjun Zhang, Xinyuan Chen, Yaohui Wang, Yue Lu, Yu Qiao},
      year={2023},
      eprint={2312.12232},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
