# FlowDreamer of Threestudio

This is the official implementation of **FlowDreamer: Exploring High Fidelity Text-to-3D Generation Via Rectified Flow**.

### [Project Page](https://vlislab22.github.io/FlowDreamer/) | [Arxiv Paper](https://arxiv.org/abs/2408.05008v3)
<!-- ![FlowDreamer Cover](https://github.com/cyjdlhy/assets/blob/main/FlowDreamer/cover.png)  
![FlowDreamer Video Demo](https://youtu.be/NCw2Qi0zoIk?si=xJamrWwk3yaULKFj) -->
### Installation

To get started with FlowDreamer, follow the installation instructions below:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/cyjdlhy/FlowDreamer_threestudio.git
    cd FlowDreamer_threestudio
    ```

2. **Create a Conda Environment**

    ```bash
    conda create -n FlowDreamer_three python=3.9.16 cudatoolkit=11.8
    conda activate FlowDreamer_three
    ```

3. **Install Dependencies**
  
  If you encounter issues while setting up the environment, please refer to [Threestudio](https://github.com/threestudio-project/threestudio). However, please note that the versions of `diffusers` and `transformers` should be installed using those specified in the `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```


4. **Download the Model**

    - Modify the `pretrained_model_name_or_path` path in the `configs/ucm.yaml` and `configs/vfds.yaml` file to point to Stable Diffusion 3 model (e.g., [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)).
    
5. **Run**

    ```bash
    bash train.sh
    ```

---

### Acknowledgements

This project is built upon the work of several excellent research projects and open-source contributions. A big thank you to all the authors for sharing their work!

- [rectified_flow_prior](https://github.com/yangxiaofeng/rectified_flow_prior.git)
- [threestudio](https://github.com/threestudio-project/threestudio)

---

### Citation

If you find this project useful in your research, please consider citing our paper:

```bibtex
@article{li2024flowdreamer,
  title={Flowdreamer: Exploring high fidelity text-to-3d generation via rectified flow},
  author={Li, Hangyu and Chu, Xiangxiang and Shi, Dingyuan and Lin, Wang},
  journal={arXiv preprint arXiv:2408.05008},
  year={2024}
}