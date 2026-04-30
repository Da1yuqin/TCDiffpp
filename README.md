# TCDiff++
Code for our paper "TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography". 


<p align="center">
  <em>✨ If you like this project, feel free to give TCDiff++ a star ✨.</em>
</p>

<p align="center">
  <img height="260" src="assets/sample.gif" />
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2506.18671">
    <img src="https://img.shields.io/badge/Paper-blue?style=for-the-badge"/>
  </a>
  <a href="https://da1yuqin.github.io/TCDiffpp.website/">
    <img src="https://img.shields.io/badge/Project Page-green?style=for-the-badge"/>
  </a>
  <!-- <a href="https://da1yuqin.github.io/TCDiffpp.website/">
    <img src="https://img.shields.io/badge/huggingface-orange?style=for-the-badge"/>
  </a> -->
  <a href="https://wanluzhu.github.io/TCDiffusion/">
    <img src="https://img.shields.io/badge/Previous TCDiff-purple?style=for-the-badge"/>
  </a>
</p>

## 🔥 News
- **2025.11.12** 🎉 Our work **TCDiff++** has been officially accepted by **IJCV 2025**
- **2025.09.28** 🚀 We have released the **TCDiff++ codebase**. The open-source implementation provides:  
  - End-to-end **trajectory-controllable diffusion framework** for group choreography.  
  - Support for both **Mamba-SSM** and **Transformer-based** backbones (via `--use_ssm` option).  
  - Scripts for **data preprocessing** on [AIOZ-GDance](https://github.com/aioz-ai/AIOZ-GDANCE).  
  - Training and evaluation pipelines with **long-duration group dance generation**.  
  - Automated **Blender visualization pipeline** for rendering high-quality 3D animations.  
- **2025.06** 📄 The preprint of **TCDiff++** is available on [arXiv](https://arxiv.org/pdf/2506.18671).  
- **2024.03** 🪄 Our previous work **[TCDiff](https://wanluzhu.github.io/TCDiffusion/)** was released, pioneering trajectory-controllable diffusion for harmonious group choreography and laying the foundation for TCDiff++.  



## Environment Setup

* To set up the environment, our environment follows [EDGE](https://github.com/stanford-tml/edge), and is configured on top of it as follows:

```bash
# Create a new conda environment
conda create -n tcdiffpp python=3.9
conda activate tcdiffpp

# Install PyTorch with CUDA support
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install additional dependencies
pip install mamba-ssm
conda install packaging

# Configure and install PyTorch3D
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/
conda install -c fvcore -c iopath -c conda-forge fvcore iopath 
conda install pytorch3d

# Install remaining requirements
pip install accelerate 
pip install librosa 
pip install matplotlib
pip install p_tqdm
```
* On certain hardware configurations, setting up an environment with SSM may encounter issues. Alternatively, you can use an environment without SSM and set the parameter `use_ssm` to `False`, which allows you to use the Transformer-based framework instead.
* We found that on certain devices, the script `create_dataset.py` has specific version requirements for NumPy (1.26.1 or 2.0.1), which may cause conflicts when setting up the environment (1.24.1) for `train.py` and `test.py`. The following actions may help resolve these issues:


```bash
pip install -U scipy
pip install -U librosa
pip install -U numpy
pip install numpy==x.x.x
```

# Data Preprocess
1. Please download AIOZ-GDance from [here](https://github.com/aioz-ai/AIOZ-GDANCE) and place it in the `./data/AIOZ_Dataset` path.


2. Run the Preprocessing Script:
```bash
cd data/
python create_dataset.py
```


# Training
To train the model, use the following command:

    accelerate launch train.py --use_ssm True
💡 Note: If you encounter difficulties setting up the mamba environment, you can set --use_ssm False. This will switch to the Transformer-based framework and remove the dependency on mamba-ssm.


# Generate results
To generate results using the trained model, run:

    python train.py --mode "val" --use_ssm True
💡 Note: Similarly, if the mamba environment is hard to configure, you can set --use_ssm False to avoid relying on SSM.

# Long group dance generation
To perform Long-duration generation, execute:

    python long_generation.py --required_dancer_num 4 --genre Electronic --use_ssm True
💡 Note: The --use_ssm setting should be kept consistent with the configuration you used during training. If you had to disable SSM due to environment issues, make sure to set --use_ssm False here as well.

# Visulization in Blender
We developed automated scripts to transform the generated SMPL motion data into beautiful 3D animations rendered in Blender, replicating the high-quality visuals featured on our project page. The entire rendering pipeline, from data preparation to Blender rendering, is fully scripted for ease of use and reproducibility. For detailed steps, please refer to the `Blender_Visulization/` Rendering Pipeline documentation.  ✨ Your star is the greatest encouragement for our work. ✨
![vis](assets/Blender_vis.png)


## Acknowledgment
The concept of TCDiff is inspired by solo-dancer generation model [EDGE](https://github.com/Stanford-TML/EDGE) and [Mamba](https://github.com/state-spaces/mamba).
We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

# Citation
```
@article{dai2025tcdiff++,
  title={TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography},
  author={Dai, Yuqin and Zhu, Wanlu and Li, Ronghui and Li, Xiu and Zhang, Zhenyu and Li, Jun and Yang, Jian},
  journal={arXiv preprint arXiv:2506.18671},
  year={2025}
}
@inproceedings{dai2025harmonious,
  title={Harmonious Music-driven Group Choreography with Trajectory-Controllable Diffusion},
  author={Dai, Yuqin and Zhu, Wanlu and Li, Ronghui and Ren, Zeping and Zhou, Xiangzheng and Ying, Jixuan and Li, Jun and Yang, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2645--2653},
  year={2025}
}
```
