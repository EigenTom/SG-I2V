<p align="center">
  <h2 align="center">SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation</h2>
  <p align="center">
    <a href=https://kmcode1.github.io/><strong>Koichi Namekata</strong></a><sup>1</sup>
    ·
    <a href=https://sherwinbahmani.github.io/><strong>Sherwin Bahmani</strong></a><sup>1,2</sup>
    ·
    <a href=https://wuziyi616.github.io/><strong>Ziyi Wu</strong></a><sup>1,2</sup>
    ·
    <a href=https://yashkant.github.io/><strong>Yash Kant</strong></a><sup>1,2</sup>
    ·
    <a href=https://www.gilitschenski.org/igor/><strong>Igor Gilitschenski</strong></a><sup>1,2</sup>
    ·
    <a href=https://davidlindell.com/><strong>David B. Lindell</strong></a><sup>1,2</sup>
</p>

<p align="center"><strong></strong></a>
<p align="center">
    <sup>1</sup>University of Toronto · <sup>2</sup>Vector Institute
</p>
   <h3 align="center">

   [![arXiv](https://img.shields.io/badge/arXiv-2411.04989-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2411.04989) [![ProjectPage](https://img.shields.io/badge/Project_Page-SG--I2V-blue)](https://kmcode1.github.io/Projects/SG-I2V/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./assets/sgi2v480x4.gif" width="100%">
  </a>
</p>

## 💡 TL;DR
Given a set of bounding boxes with associated trajectories, our framework enables object and camera motion control in image-to-video generation by leveraging the knowledge present in a pre-trained image-to-video diffusion model. Our method is self-guided, offering zero-shot trajectory control without fine-tuning or relying on external knowledge.

## 🔧 Setup
The code has been tested on:

- Ubuntu 22.04.5 LTS, Python 3.12.4, CUDA 12.4, NVIDIA RTX A6000 48GB

### Repository

```
# clone the github repo
git clone https://github.com/Kmcode1/SG-I2V.git
cd SG-I2V
```

### Installation

Create a conda environment and install PyTorch:

```
conda create -n sgi2v python=3.12.4
conda activate sgi2v
conda install pytorch=2.3.1 torchvision=0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install packages:
```
pip install -r requirements.txt
```


## :paintbrush: Usage

  ### Quick start with a notebook
  You can run ```demo.ipynb```, which contains all the implementations (along with a light explanation) of our pipeline.

  ### Reproducing qualitative results

  Alternatively, you can generate example videos demonstrated on the project website by running:

  ```
  python inference.py --input_dir <input_path> --output_dir <output_path>
  ```
  An example command that produces the same result as the notebook is ```CUDA_VISIBLE_DEVICES=0 python inference.py --input_dir ./examples/111 --output_dir ./output```. For convenience, we have provided a shell script, where it generates all the examples by running ```sh ./inference.sh```.
  
  For the input format of examples, please refer to ```read_condition(input_dir, config)``` in ```inference.py``` for more details. Briefly, each example folder contains the first frame image (```img.png```) and trajectory conditions (```traj.npy```), where the trajectory conditions are encoded by the top-left/bottom-right coordinates of each bounding box + positions of its center coordinate across frames. 

  ### Reproducing quantitative results
  
  We are currently working on releasing evaluation codes.
  

## ✏️ Acknowledgement
Our implementation is partially inspired by <a href="https://github.com/showlab/DragAnything">DragAnything</a> and <a href="https://github.com/arthur-qiu/FreeTraj">FreeTraj</a>. We thank the authors for their open-source contributions.<br>

## 📖 Citation

If you find our paper and code useful, please cite us:

```bib
@article{namekata2024sgi2v,
  author = {Namekata, Koichi and Bahmani, Sherwin and Wu, Ziyi and Kant, Yash and Gilitschenski, Igor and Lindell, David B.},
  title = {SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation},
  journal = {arXiv preprint arXiv:2411.04989},
  year = {2024},
}
