# Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model
The source code for "Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model" by Lingjun Zhang, Xinyuan Chen, Yaohui Wang, Yu Qiao, and Yue Lu. <br>
<img src="/pics/teaser.png" width="800px">
# Getting Start
## Installation
```
git clone
cd 
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

# How to run
## 1. Prepare Sketch Images
You may prepare the sketch image yourself (as shown in the first row of the image below) along with the corresponding bounding box text file (please use the ICDAR2013 or ICDAR2015 dataset annotation format), or utilize our sketch image synthesis code. The synthesis code can be executed with the following command:
```
python controlnet_util/synthtext.py
```
Before running the synthesis code, please make the necessary modifications to the "controlnet_util/Textgen/text_cfg.yaml" file, following the instructions provided in the file comments. <br>
<img src="/pics/teaser.png" width="800px">
## 2. Prepare Prompts
You may prepare a txt file of prompts.
## 3. Modify Configurations
Remember to make the necessary modifications to the "configs/control_gen.yaml", following the instructions provided in the file comments.
## 4. Run
```
python predict.py
```
