# multimodal-speech-emotion


## This repository contains the source code used in the following paper,

**Multimodal Speech Emotion Recognition using Audio and Text**, IEEE SLT-18, <a href="https://arxiv.org/abs/1810.04635">[paper]</a>

----------

### [requirements]
	tensorflow==1.4 (tested on cuda-8.0, cudnn-6.0)
	python==2.7
	scikit-learn==0.20.0
	nltk==3.3


### [download data corpus]
- IEMOCAP <a href="https://sail.usc.edu/iemocap/">[link]</a>
<a href="https://link.springer.com/article/10.1007/s10579-008-9076-6">[paper]</a>

### [source code]
- repository contains code for following models
	 > Audio Recurrent Encoder (ARE) <br>
	 > Text Recurrent Encoder (TRE) <br>
	 > Multimodal Dual Recurrent Encoder (MDRE) <br>
	 > Multimodal Dual Recurrent Encoder with Attention (MDREA) <br>

----------

### [training]
- refer "reference_script.sh"
- fianl result will be stored in "./TEST_run_result.txt" <br>


----------


### [cite]
- Please cite our paper, when you use our code | model

  > @article{yoon2018multimodal, <br>
  >  title={Multimodal Speech Emotion Recognition Using Audio and Text}, <br>
  >  author={Yoon, Seunghyun and Byun, Seokhyun and Jung, Kyomin}, <br>
  >  journal={arXiv preprint arXiv:1810.04635}, <br>
  >  year={2018} <br>
  > }
