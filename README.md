<div align="center">

# Deep Residual Echo State Networks

[![arXiv](https://img.shields.io/badge/arXiv-2508.21172-b31b1b.svg)](https://arxiv.org/abs/2508.21172)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2508.21172)
[![DOI](https://img.shields.io/badge/DOI-10.1109/TNNLS.2026.3718377-blue.svg)](https://doi.org/10.1109/TNNLS.2026.3718377)

![code-quality](https://github.com/nennomp/deepresesn/actions/workflows/code-quality.yml/badge.svg)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/nennomp/deepresesn)

</div>

This repository contains the official code for the paper:

```
Deep Residual Echo State Networks: exploring residual orthogonal connections in untrained Recurrent Neural Networks,
Matteo Pinna, Andrea Ceni, Claudio Gallicchio,
IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2026.
```

## Abstract
Echo State Networks (ESNs) are a particular type of untrained Recurrent Neural Networks (RNNs) within the Reservoir Computing (RC) framework, popular for their fast and efficient learning. However, traditional ESNs often struggle with long-term information processing. In this paper, we introduce a novel class of deep untrained RNNs based on temporal residual connections, called Deep Residual Echo State Networks (DeepResESNs). We show that leveraging a hierarchy of untrained residual recurrent layers significantly boosts memory capacity and long-term temporal modeling. For the temporal residual connections, we consider different orthogonal configurations, including randomly generated and fixed-structure, and study their effect on network dynamics. A thorough mathematical analysis outlines necessary and sufficient conditions to ensure stable dynamics within DeepResESN. Empirically, the proposed approach consistently outperforms traditional shallow and deep RC on a variety of time series tasks. Overall, DeepResESN offers a promising approach for designing hierarchical ESNs with better prediction accuracy on long sequences, without sacrificing the computational advantages that make RC attractive.

<div align="center">
<img src="assets/figure-1.png?raw=true" alt="Model" title="Model">
<br>
</div>
<figcaption><em>
<strong>Architecture of the proposed DeepResESN.</strong> (a) Structure of residual reservoir layer in a DeepResESN, modulated by an input and recurrent weight matrix (show in blue) and an orthogonal matrix (shown in purple). (b) Complete illustration of a DeepResESN architecture. The readout is the only trainable components of the model, and may be trained via closed-form solutions.
</em></figcaption>

## Setup
To install the required dependencies:
```
conda create -n deepresesn python=3.12
conda activate deepresesn
pip install -e .
```

## Citation
If you use the model or code in this repository, consider citing our paper:
```
@article{pinna2025deep,
  title={Deep Residual Echo State Networks: exploring residual orthogonal connections in untrained Recurrent Neural Networks},
  author={Pinna, Matteo and Ceni, Andrea and Gallicchio, Claudio},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026}
}
```
