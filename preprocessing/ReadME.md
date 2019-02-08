# preprocessing

**requirements (pre-download)**
> (IEMOCAP dataset) <br>
> multimodal-speech-emotion/data/raw/IEMOCAP_full_release <br>

> (pre-trained word embedding) <br>
> multimodal-speech-emotion/data/raw/embedding/glove.840B.300d.txt <br>


**process** <br>
> (run following codes) <br>
> IEMOCAP_00_extract_label_transcription.ipynb <br>
> IEMOCAP_01_wav_to_feature.ipynb <br>
> IEMOCAP_NLP_01_Transcription_to_Index.ipynb <br>
> IEMOCAP_02_to_four_category.ipynb <br>
> IEMOCAP_03_generate_train_dev_test_data.ipynb <br>
> IEMOCAP_NLP_04_Prepare_Glove.ipynb <br>


**description**

IEMOCAP_NLP_01_Transcription_to_Index.ipynb
* extract transcription and label from data
   
[ General ]
1. IEMOCAP_extract_label_transcription.ipynb

[ Audio ]
1. IEMOCAP_01_wav_to_feature.ipynb 
2. IEMOCAP_02_to_four_category.ipynb 
3. IEMOCAP_03_generate_train_dev_test_data.ipynb 
   
[ Text ]  
1. IEMOCAP_NLP_01_Transcription_to_Index.ipynb 
2. IEMOCAP_02_to_four_category.ipynb 
3. IEMOCAP_03_generate_train_dev_test_data.ipynb 
4. IEMOCAP_NLP_04_Prepare_Glove.ipynb 
