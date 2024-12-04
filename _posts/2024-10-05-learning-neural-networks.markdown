---
title:  "Learning Neural Nets"
description: >-
  Learning the foundational mathematical framework for AI, Neural Networks, ML and DL.
author: lso
date:   2024-10-05 00:08:03 +0200
categories: [Blogging, Tutorial]
tags: [neuralnetworks]
pin: true
---

![AI/ML](../../assets/images/ai-ml-basis.png)

## Linear Algebra

| Link | Description |
|------|-------------|
| [Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | 3Blue1Brown Series on Linear Algebra |
| [Mathematics for Machine Learning](https://www.youtube.com/watch?v=0z6AhrOSrRs) | Imperial College London Course on  Mathematics for Machine Learning |
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | Anthropic's Mathematical Framework for Transformers |
| [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations/tree/master) | Machine Learning Foundations |

## Calculus

| Link                                                                                      | Description                                      |
|-------------------------------------------------------------------------------------------|--------------------------------------------------|
| [Calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)                             | 3Blue1Brown's series on calculus.    |
| [MIT Calculus 1A: Differentiation](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+18.01.1x+2T2019/course/)       | Calculus 1A by MIT. |
| [Understanding Calculus: Problems, Solutions, and Tips](https://www.thegreatcourses.com/courses/understanding-calculus-problems-solutions-and-tips) | A course offering various problems, solutions, and tips for understanding calculus.   |
| [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations/tree/master) | Machine Learning Foundations |

## Differentiable Programming

| Link                                                                                      | Description                                      |
|-------------------------------------------------------------------------------------------|--------------------------------------------------|
| [Google Deepmind: The Elements of Differentiable Programming](https://arxiv.org/pdf/2403.14606) | The mathematical frameworks that underpins "Differentiable Programming"     |



## Neural Networks

| Link                                                                                                      | Description                                              |
|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [Before Language Models - N-Ggram](https://en.wikipedia.org/wiki/Word_n-gram_language_model)               | An article on the Wikipedia page for Word n-gram language model. |
| [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)                | 3Blue1Brown's series on neural networks.         |
| [Karpathy's Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)                    | Andrej Karpathy's series on neural networks |
| [Build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)                                      | Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out.   |
| [Umar Jamil's youtube channel](https://www.youtube.com/@umarjamilai)                                      | Umar Jamil's youtube channel.   |
| [Shawhin Talebi's youtube channel](https://www.youtube.com/@ShawhinTalebi) | Shawhin Talebi's youtube channel. |

## Architectures

| **Domain**              | **Problem**                        | **Architecture**                                                                                   |
|--------------------------|------------------------------------|---------------------------------------------------------------------------------------------------|
| **Computer Vision**      | Image classification              | CNNs, ResNet, EfficientNet, Vision Transformers (ViT)                                             |
|                          | Object detection                  | YOLO, Faster R-CNN, RetinaNet, DETR                                                              |
|                          | Image segmentation                | U-Net, Mask R-CNN, DeepLabV3                                                                      |
|                          | Image generation                  | GANs, Variational Autoencoders (VAEs)                                                            |
|                          | Style transfer                    | Neural Style Transfer Networks                                                                    |
| **Natural Language Processing (NLP)** | Text classification               | RNNs, BiLSTMs, Transformers, BERT                                                                 |
|                          | Sentiment analysis                | RNNs, BiLSTMs, Transformers, BERT                                                                 |
|                          | Machine translation               | Seq2Seq Models, Transformers, MarianMT, T5                                                       |
|                          | Text summarization                | Transformers, BART, Pegasus                                                                       |
|                          | Question answering                | BERT, RoBERTa, ALBERT                                                                             |
|                          | Text generation                   | GPT, LLaMA                                                                                       |
| **Speech Processing**    | Speech recognition                | RNNs with CTC Loss, Wav2Vec 2.0, Conformer                                                       |
|                          | Speech synthesis                  | Tacotron 2, WaveNet, VITS                                                                         |
|                          | Speaker identification            | CNNs, ResNet-based models                                                                         |
| **Time-Series Analysis** | Forecasting                       | LSTMs, GRUs, Temporal Fusion Transformer                                                         |
|                          | Anomaly detection                 | Autoencoders, LSTMs                                                                               |
|                          | Activity recognition              | CNNs, RNNs, Temporal Convolutional Networks (TCNs)                                               |
| **Reinforcement Learning** | Game playing                     | Deep Q-Networks (DQNs), AlphaZero, PPO                                                           |
|                          | Robotics control                  | Policy Gradient Methods, SAC, TD3                                                                |
|                          | Strategy optimization             | A3C, PPO                                                                                         |
| **Generative Modeling**  | Text-to-image generation          | Stable Diffusion, DALL-E                                                                          |
|                          | Data augmentation                 | GANs, VAEs                                                                                       |
|                          | Synthetic data generation         | GANs, Diffusion Models                                                                            |
| **Medical Applications** | Disease detection                 | CNNs, Vision Transformers (ViT)                                                                  |
|                          | Medical image segmentation        | U-Net, SegNet                                                                                    |
|                          | Drug discovery                    | Graph Neural Networks (GNNs), Transformer-based models (e.g., MolBERT)                           |
| **Graph-Based Problems** | Social network analysis           | Graph Neural Networks (GNNs), Graph Convolutional Networks (GCNs)                                |
|                          | Knowledge graph completion        | Graph Attention Networks (GATs), TransE                                                          |
|                          | Molecular structure prediction    | Message Passing Neural Networks (MPNNs)                                                          |
| **Recommendation Systems** | Product recommendation          | Collaborative Filtering Models, Autoencoders, Neural Collaborative Filtering (NCF), Transformers |
|                          | Content personalization           | Collaborative Filtering Models, Autoencoders, Transformers                                       |
|                          | Collaborative filtering           | Collaborative Filtering Models, Neural Collaborative Filtering (NCF)                             |
| **Control Systems**      | Autonomous driving                | Convolutional Networks, Reinforcement Learning with sensor integration                           |
|                          | Industrial process optimization   | Recurrent Neural Networks, Reinforcement Learning                                                |
| **Multimodal Learning**  | Image and text alignment          | CLIP, ViLBERT                                                                                   |
|                          | Audio-visual synchronization      | Multimodal Transformers, Hybrid CNN-RNN Architectures                                            |
