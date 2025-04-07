# Online Language Splatting

[Saimouli Katragadda](https://saimouli.github.io/), [Cho-Ying Wu](https://choyingw.github.io), [Yuliang Guo†](https://yuliangguo.github.io/), [Xinyu Huang](https://scholar.google.com/citations?user=cL4bNBwAAAAJ&hl=en), [Guoquan Huang](https://udel.edu/~ghuang/), [Liu Ren](https://sites.google.com/site/liurenshomepage/)  
(† indicates corresponding author)  
[**Webpage**](https://saimouli.github.io/onlineLang/) | [**Paper**](https://arxiv.org/pdf/2503.09447) | [**Video**](https://www.youtube.com/watch?v=GIldru2006k&feature=youtu.be)  
**Preprocessed Dataset**: [DropBox]()  
**Pretrained Models**: [DropBox]()

<table>
  <tr>
    <td align="center">
      <img src="media/langslam_sofa.gif" width="400px" alt="Sofa Demo"/><br/>
      <b>Sofa</b>
    </td>
    <td align="center">
      <img src="media/langslam_rug.gif" width="400px" alt="Rug Demo"/><br/>
      <b>Rug</b>
    </td>
  </tr>
</table>




---

## 🔔 Note

With minor modifications to the SLAM pipeline, our models can be integrated into Gaussian Splatting-based SLAM systems.

This release includes a clean implementation of our core model, along with disentangled optimization available under submodules for reference. Note that integrating disentangled optimization requires a high-performance GPU, which we plan to support in future releases.

We’re actively working on improving the pipeline for higher speed, larger-scale datasets, and broader compatibility — stay tuned for upcoming updates!

---


## 🚀 Getting Started

### 📦 Dataset

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip
unzip vmap.zip
```
## 🛠️ Installation
```bash
git clone https://github.com/saimouli/GaussianGripMapping.git --recursive
cd GaussianGripMapping
```
Setup the environment.

```bash
conda env create -f environment.yaml
conda activate LangGS
```

💬 Language Model Setup

```bash
cd langauge/sed/open_clip && make install
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Download language model weights from 
```
https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view?usp=drive_link

```
Edit ```language/configs/convnextB_768.yaml``` and Set the  ```WEIGHTS``` to the path of the downloaded language model weights

```bash
python create_lang_model.py
```

# 🧠 Language
To test language feature
```bash
python3 language/language_features.py --high-res-model "high_res_71_indoor.ckpt" --lang-model "seg_clip_model_l.pth" --input "test.png" --query-text "checkerboard"
```

# 🧭 Running the Pipeline

Edit base_config.yaml file to load `auto_ckpt_path` to load generalized autoencoder. `lang_model_path` to point to the language feature map model weights and `hr_ckpt_path` to point to the high resolution module weights.

for room0.yaml edit `dataset_path` to point to the room0 dataset and `online_ckpt_path` to where you want the checkpoint to be saved.

### To Run ▶️ 2-Stage Pipeline
In base_config.yaml point `auto_ckpt_path` and `hr_ckpt_path` to the respective files and in room0.yaml set `single_stage_ae` to `False`.

### To Run ▶️ 1-Stage Pipeline
In room0.yaml point `auto_ckpt_path` to the cross data generalization file and set `single_stage_ae` to `True`.

```bash
python3 slam.py --config configs/rgbd/replicav2/room0.yaml
```

# Evaluate
🔖 Create Labels
```bash
python3 eval/create_replica_labels.py
```

## ✅ Evaluate 2-Stage Pipeline

To evaluate 2 stage
```bash
python3 eval/evaluate_onlinelangslam.py
```
## ✅ Evaluate 1-Stage Pipeline
To evaluate cross data genenarizable 
```bash
python3 eval/evaluate_langslam.py
```
## 🧱 3D Evaluation
⚠️ Note: in each .py file, please read the comment and change path variables that match your local.

Prepare colorized GT by running 
```bash
cd eval/tsdf_fusion
python3 save_semantic_colors_gt.py
```

To reconstruct TSDF for groundtruth, run
```bash
python3 dim3_recon_gt.py
```

```bash
cd PytorchEMD; python3 setup.py
```
copy the compiled .so file to the tsdf-fusion folder (move one level up)

▶️ Run 3D Evaluation
LangSlam
```bash
python3 3d_evaluation_and_visualize_langslam_dim15.py
```

LangSplat
```bash
python3 3d_evaluation_and_visualize_langsplat.py
```

🧪 Training
### To train your own AE on your domain for 1-stage
```bash
python3 language/autoencoder/train_encoder_light.py
```

We provide sample evaluations for room0 here:

# 🧬 Reprodicibility
There might be minor differences between the released version and the results in the paper. Please bear in mind that multi-process performance has some randomness due to GPU utilisation. We run all our experiments on an RTX A4500 GPU, and the performance may differ when running with a different GPU.

# 🙏 Acknowledgement
This work incorporates many open-source codes. We extend our gratitude to the authors of the software.

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization
](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [SED](https://github.com/xb534/SED)
- [MonoGS](https://github.com/muskie82/MonoGS)
- [LangSplat](https://github.com/minghanqin/LangSplat)

# 📖 Citation
If you find this work helpful, please consider citing us:

```bibtex
@inproceedings{katragadda2025_onlinelang,
  title     = {{O}nline {L}anguage {S}platting},
  author    = {Saimouli Katragadda and Cho-Ying Wu and Yuliang Guo and Xinyu Huang and Guoquan Huang and Liu Ren},
  booktitle = {arXiv},
  year      = {2025}
}
```











