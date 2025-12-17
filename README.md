# InfoDCL: Informative Noise Enhanced Diffusion Based Contrastive Learning

Here is the official PyTorch implementation for the paper **"InfoDCL: Informative Noise Enhanced Diffusion Based Contrastive Learning"**, which has been accepted by **KDD '26**.

This project proposes a novel framework **InfoDCL** to address the critical limitations of **random view construction** and **insufficient semantic information** in sparse recommendation scenarios. InfoDCL employs a diffusion-based process that integrates auxiliary semantics to generate authentic user preference views, and utilizes a collaborative training strategy to transform the interference between generation and preference learning into mutual collaboration.

**Authors**: Xufeng Liang, Zhida Qin, Chong Zhang, Tianyu Huang, and Gangyi Ding.
**Affiliation**: Beijing Institute of Technology, Xi'an Jiaotong University.

## Architecture

The overall architecture of InfoDCL consists of three main parts: **Informative Noise Generation**, **Semantics Enhanced Contrastive Learning**, and **Collaborative Training Objective Strategy**.

![GitHub Logo](main.png)


## Requirements

The code is implemented using **PyTorch**. The mainly required packages are listed below:

```bash
python>=3.8
torch>=1.10.0
numpy>=1.20.3
scipy>=1.6.2
torchdiffeq>=0.2.0  # For ODE solvers
```

## Usage

<ol> <li>Data Preparation: Download the datasets from <a href="https://jmcauley.ucsd.edu/data/amazon/">Amazon Review Data</a> and MovieLens, and place them in the <code>dataset/</code> directory.</li>  <li>Training: Run the main script to train and evaluate the model:</li> </ol>

```bash
python main.py --model TGODE --dataset Beauty
```

## Implemented Models

<table class="table table-hover table-bordered"> <tr> <th>Model</th>         <th>Paper</th>      <th>Type</th>   <th>Code</th> </tr> <tr> <td scope="row">TGODE</td> <td>Fu et al. <a href="https://doi.org/10.1145/3711896.3737156" target="_blank">Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs</a>, KDD '25. </td> <td>GNN + ODE + Diffusion</td> <td><a href="https://github.com/fhy99/TGODE">PyTorch</a> </td> </tr> </table>

## Related Datasets

We conduct extensive experiments on five real-world datasets. The statistics are summarized below :

| **Datasets** | **ML-1M** | **Office** | **Baby** | **Taobao** | **Electronics** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#Users** | 6040 | 4,905 | 19,445 | 12,539 | 32,886 |
| **#Items** | 3706 | 2,420 | 7,050 | 8,735 | 52,974 |
| **#Int.** | 1,000,209 | 53,258 | 159,669 | 83,648 | 337,837 |
| **Sparsity** | 95.53% | 99.55% | 99.88% | 99.92% | 99.69% |

## Reference

If you find this repo helpful to your research, please cite our paper :

```BibTeX
@inproceedings{fu2025time,
  title={Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs},
  author={Fu, Haoyan and Qin, Zhida and Yang, Shixiao and Zhang, Haoyao and Lu, Bin and Li, Shuang and Huang, Tianyu and Lui, John CS},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={637--648},
  year={2025}
}

```
