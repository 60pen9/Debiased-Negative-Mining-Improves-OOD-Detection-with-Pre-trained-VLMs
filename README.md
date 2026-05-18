# Debiased-Negative-Mining-Improves-OOD-Detection-with-Pre-trained-VLMs
KDD2026

This is the code for the paper "A Closer Look at Negative Label Guided Out-of-distribution Detection with Pre-trained Vision-Language Models" (ICML 2026).

Abstract: *Aiming at identifying unexpected inputs from unknown classes, out-of-distribution (OOD) detection has emerged as a pivotal approach to enhancing the reliability of machine learning models.
This paper focuses on the burgeoning paradigm of post-hoc OOD detection with pre-trained vision-language models (VLMs), where a popular pipeline is to detect OOD inputs by examining their affinities between ID labels and negative labels, i.e., those semantically different from ID labels.
Due to the unavailability of target OOD labels, existing works predominantly rely on heuristic rules to mine negative labels from unlabeled wild corpus data. 
Despite the empirical success, we argue that the power of VLM-based OOD detection has yet to be fully unleashed since the notorious false negative problem is far from addressed in the literature.
With this motivation, we are interested in addressing the challenge of mining true negative labels for OOD scoring.
To this end, we develop a theoretical framework for correcting the sampling bias of negatives labels by indirectly approximating the distribution of negative labels.
Perhaps surprisingly, we show that the debiased negative mining can be naturally converted into Monte-Carlo sampling based on ID labels and the unlabeled wild corpus data.
Extensive experiments empirically manifest that our method establishes a new state-of-the-art in a variety of OOD detection setups.*
