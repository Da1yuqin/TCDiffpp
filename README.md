# TCDiffpp
This is the official code for our paper: "TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography". 

The code is being released soon!

[<a href="https://da1yuqin.github.io/TCDiffpp.website/"><strong>Project Page</strong></a>]

![model](Figs/pipline.jpg)
Our end-to-end TCDiff++ framework comprises two key components: the Group Dance Decoder (GDD) and the Footwork Adaptor (FA). The GDD initially generates a raw motion sequence without trajectory overlap based on the given music. Subsequently, the FA refines the foot movements by leveraging the positional information of the raw motion, producing an adapted motion with improved footstep actions to reduce foot sliding. Finally, the adapted footstep movements are incorporated into the raw motion, yielding a harmonious dance sequence with stable footwork and fewer dancer collisions.



# Citation
```
@article{dai2025tcdiff++,
  title={TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography},
  author={Dai, Yuqin and Zhu, Wanlu and Li, Ronghui and Li, Xiu and Zhang, Zhenyu and Li, Jun and Yang, Jian},
  journal={arXiv preprint arXiv:2506.18671},
  year={2025}
}
@article{dai2024harmonious,
  title={Harmonious Group Choreography with Trajectory-Controllable Diffusion},
  author={Dai, Yuqin and Zhu, Wanlu and Li, Ronghui and Ren, Zeping and Zhou, Xiangzheng and Li, Xiu and Li, Jun and Yang, Jian},
  journal={arXiv preprint arXiv:2403.06189},
  year={2024}
}
```
