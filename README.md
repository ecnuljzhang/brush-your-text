# Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model
The source code for "Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model" by Lingjun Zhang, Xinyuan Chen, Yaohui Wang, Yu Qiao, and Yue Lu. <br>
<img src="/pics/results.png" width="800px">
# Getting Start
## Environment
```
conda create -n difftext python=3.8.5
conda activate difftext
pip install -r requirements.txt
```

## Checkpoints
please refer to https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0 to install the dependencies of deformable convolution.

## Dataset
The directory hierarchy is shown as follows: 
```
LITST-Dataset
|--- color masks
|          |--- color_1.jpg
|          |--- ...
|--- std_font
|          |--- English_1.jpg
|          |--- ...
|--- train
|          |--- font_style_1
|          |--- font_style_2
|                  |--- Hindi_1.jpg
|                  |--- ...
|          |--- ...
|--- valid
|          |--- font_style_3
|          |--- font_style_4
|                  |--- Hindi.jpg
|                  |--- ...
|          |--- ...
```

# How to run
## 1. Prepare dataset
Download proposed [dataset](https://drive.google.com/file/d/1K2evs9p3VLeKGWgPJkV-AahD1J5NOZLp/view?usp=sharing). 
## 2. Train
You can train on proposed dataset with the following code:
```
python train_proposed.py
```
Or you can also download the pretrained [model](https://drive.google.com/file/d/1XXwCE7tmMyELIKp4cRAo7b-DG0LN2I7z/view?usp=sharing).
## 3. Test
```
python test.py
```
You can define font_style_path and im_src, im_dist to get the generated characters for that particular font style.
