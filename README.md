# multimodal-speech-emotion


## This repository contains the source code used in the following paper,

**Multimodal Speech Emotion Recognition using Audio and Text**, IEEE SLT-18, <a href="https://arxiv.org/abs/1810.04635">[paper]</a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimodal-speech-emotion-recognition-using/speech-emotion-recognition-on-iemocap)](https://paperswithcode.com/sota/speech-emotion-recognition-on-iemocap?p=multimodal-speech-emotion-recognition-using)

----------

### [requirements]
	tensorflow==1.4 (tested on cuda-8.0, cudnn-6.0)
	python==2.7
	scikit-learn==0.20.0
	nltk==3.3


### [download data corpus]
- IEMOCAP <a href="https://sail.usc.edu/iemocap/">[link]</a>
<a href="https://link.springer.com/article/10.1007/s10579-008-9076-6">[paper]</a>
- download IEMOCAP data from its original web-page (license agreement is required)


### [preprocessed-data schema (our approach)]
- for the preprocessing, refer to codes in the "./preprocessing"
- If you want to download the "preprocessed corpus" from us directly, please send us an email after getting the license from IEMOCAP team.
- We cannot publish ASR-processed transcription due to the license issue (commercial API), however, we assume that it is moderately easy to extract ASR-transcripts from the audio signal by oneself. (we used google-cloud-speech-api)
- Examples
	> MFCC : MFCC features of the audio signal (ex. train_audio_mfcc.npy) <br>
	> MFCC-SEQN : valid lenght of the sequence of the audio signal (ex. train_seqN.npy)<br>
	> PROSODY : prosody features of the audio signal (ex. train_audio_prosody.npy) <br>
	> LABEL : targe label of the audio signal (ex. train_label.npy) <br> 
	> TRANS : sequences of trasnciption (indexed) of a data (ex. train_nlp_trans.npy) <br>


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
- Please cite our paper, when you use our code | model | dataset

  > @article{yoon2018multimodal, <br>
  >  title={Multimodal Speech Emotion Recognition Using Audio and Text}, <br>
  >  author={Yoon, Seunghyun and Byun, Seokhyun and Jung, Kyomin}, <br>
  >  journal={arXiv preprint arXiv:1810.04635}, <br>
  >  year={2018} <br>
  > }
