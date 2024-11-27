---
title:  "Optimizations"
description: >-
  Optimizations.
author: lso
date:   2024-10-03 11:08:03 +0200
categories: [Blogging, Tutorial]
tags: [neuralnetworks, optimizations, quantization]
pin: false
media_subpath: '/posts/20241003'
---

# Quantization

{% jupyter_notebook "../../assets/quantization.ipynb" %}

# Knowledge Distillation

Transfer the knowledge of a larger (teacher) model to a smaller (student) one.

Soft targets: train student using logits of the teacher. During training, the teacher's logits are used as targets for the student.
Synthetic data: generate synthetic data from the teacher's predictions.

# Sparsity

# Torch compiler

# References

[GPU MODE IRL 2024 Keynotes](https://www.youtube.com/watch?v=FH5wiwOyPX4&ab_channel=GPUMODE)
[Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training](https://www.youtube.com/watch?v=0VdNflU08yA)
[Knowledge Distillation](https://www.youtube.com/watch?v=FLkUOkeMd5M)
