# Infusion
Infusion: Preventing Customized Text-to-Image Diffusion from Overfitting
<img src="./static/images/teaser_small.png" width="96%" height="96%">
**[[Project Page]](https://zwl666666.github.io/infusion/)** | **[[Paper]](https://arxiv.org/abs/2404.14007)**
**Abstract:** Text-to-image (T2I) customization aims to create images that embody specific visual concepts delineated in textual descriptions
However, existing works still face a main challenge, concept overfitting. To tackle this challenge, we first analyze overfitting, categorizing it into concept-agnostic overfitting, which undermines non-customized concept knowledge, and concept-specific overfitting, which is confined to customize on limited modalities, i.e, backgrounds, layouts, styles. To evaluate the overfitting degree, we further introduce two metrics, i.e, Latent Fisher divergence and Wasserstein metric to measure the distribution changes of non-customized and customized concept respectively.
Drawing from the analysis, we propose Infusion, a T2I customization method that enables the learning of target concepts to avoid being constrained by limited training modalities, while preserving non-customized knowledge. Remarkably, Infusion achieves this feat with remarkable efficiency, requiring a mere 11KB of trained parameters. Extensive experiments also demonstrate that our approach outperforms state-of-the-art methods in both single and multi-concept customized generation. <br>

## TODOs
- [ ] Release Inference code.
- [ ] Release pretrained models.
- [ ] Release training code.
## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{zeng2024infusion,
  title={Infusion: Preventing Customized Text-to-Image Diffusion from Overfitting},
  author={Zeng, Weili and Yan, Yichao and Zhu, Qi and Chen, Zhuo and Chu, Pengzhi and Zhao, Weiming and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2404.14007},
  year={2024}
}
```
## Acknowledgement

This repo benefits from [Custom Diffusion](https://github.com/adobe-research/custom-diffusion), [Perfusion](https://github.com/ChenDarYen/Key-Locked-Rank-One-Editing-for-Text-to-Image-Personalization) Thanks for their wonderful works.
