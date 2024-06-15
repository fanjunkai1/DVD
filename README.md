# DVD: Driving-Video Dehazing with Non-Aligned Regularization for Safety Assistance (CVPR 2024)

[Junkai Fan](https://fanjunkai1.github.io/),
[Jiangwei Weng](https://wengjiangwei.github.io/),
[Kun Wang](https://github.com/w2kun/),
[Yijun Yang](https://yijun-yang.github.io/),
[Jianjun Qian](http://www.patternrecognition.asia/qian/),
[Jun Li](https://sites.google.com/view/junlineu/),
[Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN)

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2405.09996)
[![Website](figs/badge-website.svg)](https://fanjunkai1.github.io/projectpage/DVD/index.html)

This repository represents the official implementation of the paper titled "Driving-Video Dehazing with Non-Aligned Regularization for Safety Assistance".

## Video demo
To validate the stability of our video dehazing results, we present a video result captured in a real driving
environment and compare it with the latest video dehazing state-of-the-art method, MAP-Net.

https://github.com/fanjunkai1/DVD/assets/138647972/05eda045-7122-412b-87c0-8ba6a49fadc1.mp4


## Our pipeline
Our method effectively trains the video dehazing network using real-world hazy and clear videos without requiring strict alignment, resulting in high-quality results.

<img src = "figs/pipeline.png" width='840' height='300'>
