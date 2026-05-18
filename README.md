# Debiased-Negative-Mining-Improves-OOD-Detection-with-Pre-trained-VLMs

This is the code for the paper "A Closer Look at Negative Label Guided Out-of-distribution Detection with Pre-trained Vision-Language Models" (KDD2026).

Abstract: *Aiming at identifying unexpected inputs from unknown classes, out-of-distribution (OOD) detection has emerged as a pivotal approach to enhancing the reliability of machine learning models. This paper focuses on the burgeoning paradigm of post-hoc OOD detection with pre-trained vision-language models (VLMs), where a popular pipeline is to detect OOD inputs by examining their affinities between ID labels and negative labels, i.e., those semantically different from ID labels. Due to the unavailability of target OOD labels, existing works predominantly rely on heuristic rules to mine negative labels from unlabeled wild corpus data. Despite the empirical success, we argue that the power of VLM-based OOD detection has yet to be fully unleashed since the notorious false negative problem is far from addressed in the literature. With this motivation, we are interested in addressing the challenge of mining true negative labels for OOD scoring. To this end, we develop a theoretical framework for correcting the sampling bias of negatives labels by indirectly approximating the distribution of negative labels. Perhaps surprisingly, we show that the debiased negative mining can be naturally converted into Monte-Carlo sampling based on ID labels and the unlabeled wild corpus data. Extensive experiments empirically manifest that our method establishes a new state-of-the-art in a variety of OOD detection setups.*

# Dataset Preparation

## In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) 

## Out-of-distribution dataset

Following [KNN](https://arxiv.org/abs/2204.06507), we use the following 4 OOD datasets for evaluation: [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), and [Textures](https://arxiv.org/pdf/1311.3618.pdf).

Please refer to [KNN](https://github.com/deeplearning-wisc/knn-ood), download OOD datasets.

# Usage
We first need to compute the image embedding, ID class embedding, filter negative label embedding with the CLIP model by running

> python eval_ood_detection.py

and perform OOD scoring by running

> python main_online.py

