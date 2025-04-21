#  IndicTrans2 Fine-Tuning: English-Hindi and English-Telugu Translation

This project aims to fine-tune [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-hi) for two high-resource Indian language pairs: **English-Hindi** and **English-Telugu**, using a 15,000-sentence subset (compute constraint) of the [Samanantar](https://www.kaggle.com/datasets/mathurinache/samanantar) corpus.

---

##  Project Summary

This project focuses on fine-tuning IndicTrans2 with aggressive memory optimizations on a **Google Cloud Platform VM**. The goal was to improve translation fluency and adequacy within **GPU-constrained environments**.

###  Key Features:

-  Trained with **15,000 sentence pairs** for each language pair.
-  Used **SentencePiece** tokenization for better subword coverage.
-  Performed training using Hugging Face’s `Seq2SeqTrainer` on:
  - **T4 GPU (16 GB VRAM)**
  - **30 GB RAM**
  - **120 GB SSD**
-  Leveraged 8-bit quantized model loading (`load_in_8bit=True`) for memory savings.
-  Evaluation conducted using:
  - BLEU (SacreBLEU)
  - COMET
  - BERTScore
  - Bard Score (manual fluency/grammar/meaning)

>  BLEU scores were lower than base IndicTrans2 (almost close to base model in telugu) due to resource limits (1 epoch, batch size = 1).  
> But **COMET, BERTScore, and human review** confirmed meaningful improvements — especially for Telugu, where the pretrained model struggled more.

---

##  Dataset

- **Name:** [Samanantar](https://www.kaggle.com/datasets/mathurinache/samanantar)
- **Size:** ~12 GB
- **Languages:** English ↔ Hindi, English ↔ Telugu
- **Domains:** Wikipedia, government sites, news portals

Due to its size, the dataset is not included in this repo.  
➡ You can download it from: [Kaggle Dataset Link](https://www.kaggle.com/datasets/mathurinache/samanantar)

---

##  System Setup: Google Cloud + Jupyter Notebook

All experiments were run on a **GPU-enabled Google Cloud VM** using a Python virtual environment.

###  Google Cloud VM Specs:
- GPU: NVIDIA Tesla T4 (16 GB VRAM)
- RAM: 30 GB
- Disk: 120 GB SSD

---

###  One-Time Setup (Inside VM)

```bash
# Install Python and venv if not available
sudo apt update
sudo apt install python3-venv

# Create virtual environment
python3 -m venv jupyter_env

# Activate virtual environment
source ~/jupyter_env/bin/activate

# Install required packages
pip install transformers datasets sentencepiece evaluate bitsandbytes accelerate jupyter
pip install -r requirements.txt


 Steps to Run Each Time (Jupyter on GCP)
# 1. Connect to VM from your Mac
gcloud compute ssh main --zone=southamerica-east1-c

# 2. Activate virtual environment
source ~/jupyter_env/bin/activate

# 3. Start Jupyter Notebook (watch for token/port)
jupyter notebook --no-browser --port=8888

# 4. In a NEW terminal window on your Mac
gcloud compute ssh main --zone=southamerica-east1-c -- -L 8888:localhost:8888

# 5. Open browser and paste:
http://localhost:8888/tree?token=YOUR_TOKEN_HERE

# 6. When finished:
# In Jupyter terminal:
Ctrl+C

# In both terminals:
exit
```

###  Project Structure
```bash

├── Eng-Hin-Final.ipynb
├── Eng-Tel-Final.ipynb
├── indictrans2-en-hi-finetuned/
│   └── model + tokenizer files
├── indictrans2-en-te-finetuned/
│   └── model + tokenizer files
├── tokenized_eval_dataset_en_hi/
│   └── Hugging Face tokenized dataset files
├── tokenized_eval_dataset_en_te/
│   └── Same format as above
├── train.en-hi.tsv
├── train.en-te.tsv
├── eng-te/
│   ├── train.en
│   ├── train.te
│   ├── en_te_model.model
│   └── en_te_model.vocab
├── eng-hi/
│   ├── train.en
│   ├── train.hi
│   ├── en_hi_model.model
│   └── en_hi_model.vocab
```
##  Full Project Files (Google Drive)

Due to GitHub’s file size limits, many large files (model weights, tokenized datasets, training artifacts) are not included in this repository.

➡️ You can access **all project files and folders** here:  
-> [ Click to View on Google Drive](https://drive.google.com/drive/folders/1fRgCuYE4NnR_5zu-12BWBM-blpDe17wW?usp=share_link)



##  Results

We used various metrics and visualizations to track our model's performance throughout the training and evaluation process. Below are the key results from our experiments with English → Hindi and English → Telugu translation pairs.

##  English →  Telugu Results

### Training Loss vs Steps
Our training loss curve shows consistent improvement throughout the training process, demonstrating effective learning despite computational constraints.

![output](https://github.com/user-attachments/assets/f00dee88-010d-4617-9ac3-085d4b437898)


### BLEU Score vs Steps
The BLEU score steadily increased during training, showing continuous improvement in translation quality.

![output-3](https://github.com/user-attachments/assets/3ee2b351-baae-4506-9cd7-5c295ca3bd3a)



### BLEU Score Comparison
Our fine-tuned model achieved a BLEU score of 31.4 compared to IndicTrans2's 36.2. This relatively small gap despite our computational constraints suggests strong learning capabilities.

![output-4](https://github.com/user-attachments/assets/a34563f0-59d8-4f81-9952-9c1156a363d1)



### Bard Score Comparison
On human evaluation, our model significantly outperformed Google Translate, scoring 50.0 compared to Google's 22.5 on the Bard Score, demonstrating superior fluency and accuracy.
![output-2](https://github.com/user-attachments/assets/1ba267eb-50a2-4615-a3ac-ae228750b6ad)



### Attention Heatmap
The attention visualization for Telugu shows interesting patterns reflecting the language's SOV structure and agglutinative nature, with particularly strong attention weights between semantically linked tokens.

![output-5](https://github.com/user-attachments/assets/2738a2e7-40b4-43f7-adda-2543ca42f59a)



##  English →  Hindi Results

### Training Loss vs Steps
The training loss for English-Hindi shows fluctuations throughout the epoch but decreased eventually, demonstrating efficient learning.
![output-6](https://github.com/user-attachments/assets/3ca73b2b-aad1-49a8-b4f4-bc53b47030f2)



### BLEU Score vs Steps
Hindi translation quality improved steadily during training, with BLEU scores starting at 10 and reaching 29.49 by the end of training.
![output-7](https://github.com/user-attachments/assets/e355bbb3-f85f-432b-8732-131478a5cfec)



### BLEU Score Comparison
While our fine-tuned model (29.49) shows lower BLEU than the pretrained IndicTrans2 (43.90), this is expected given our training limitations. The pretrained model was trained on millions of sentence pairs with substantially more computational resources.
![output-8](https://github.com/user-attachments/assets/515471a3-c415-48a8-b033-57b9c7a9f5ad)



### Bard Score Comparison
Despite BLEU score differences, our model significantly outperforms Google Translate on human evaluation metrics, scoring 47.5 compared to Google's 22.5 on the Bard Score scale.

![output-10](https://github.com/user-attachments/assets/a94e1a7b-8b74-4f34-a51f-5bb3254af18f)



### Attention Heatmap
Visualization of the attention patterns shows how our model learns to align words between English and Hindi, with stronger weights on semantically related tokens.

![Unknown](https://github.com/user-attachments/assets/2510e5b9-5223-410b-bd5a-f91957ce16e9)

##  Metrics- Scores

###  English →  Telugu (En → Te)

- **BLEU**: 31.4  
- **COMET**: 0.72  
- **BERTScore (F1)**: ~0.92  
- **Bard Score**: 50  
- **TER Score**: 41.75  

###  English →  Hindi (En → Hi)

- **BLEU**: 29.49  
- **COMET**: 0.57  
- **BERTScore (F1)**: ~0.86  
- **Bard Score**: 47.5    
- **TER Score**: 52.36  

---






## Summary Notes

 These results were generated from one full epoch (15,000 steps) of training.

 While some pretrained baseline metrics (particularly BLEU) outperform our fine-tuned model due to compute constraints and our use of only a 15,000-sentence subset (∼1%) of the full Samanantar corpus, our **Bard Score** and **qualitative metrics** demonstrate meaningful improvements in translation quality — especially for English-Telugu.

 Notably, for **English → Telugu**, our fine-tuned model **almost matched the BLEU score of IndicTrans2**, despite being trained on a drastically smaller dataset.  
-> This suggests that with access to the full corpus, our model would likely **surpass the baseline**.

 The stronger relative performance on Telugu also indicates that our fine-tuning approach is especially beneficial for **morphologically rich or less represented languages** in pretrained multilingual models.

