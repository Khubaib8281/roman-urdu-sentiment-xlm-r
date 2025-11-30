---
library_name: transformers
tags:
- sentiment-analysis
- roman-urdu
- xlm-roberta
- classification
- multilingual
license: mit
metrics:
- accuracy
- f1
base_model:
- FacebookAI/xlm-roberta-base
pipeline_tag: text-classification
---

# Model Card for Model ID

This model performs **sentiment classification on Roman Urdu text**, labeling input sentences as **Positive, Negative, or Neutral**. It is fine-tuned from `xlm-roberta-base` and is designed to handle Roman Urdu, code-mixed Urdu-English, and social media style text.

## Try the Model
You can test the Roman Urdu Sentiment Analysis model live on Hugging Face Spaces:
[![Try on Hugging Face](https://img.shields.io/badge/🤗-Try%20it%20on%20Spaces-blue)](https://huggingface.co/spaces/Khubaib01/roman-urdu-sentiment)

## Model Details

### Model Description

This model is a **fine-tuned XLM-RoBERTa Base** transformer for sentiment analysis in Roman Urdu. It was trained on a balanced dataset of **~54k sentences**, with labels: Positive, Negative, and Neutral. The dataset includes messages from **WhatsApp chats and YouTube comments**, preprocessed to remove media, deleted messages, short messages, and pure emoji-only messages.  

- **Developed by:** Muhammad Khubaib Ahmad  
- **Model type:** XLM-RoBERTa Base (transformers)  
- **Language(s):** Roman Urdu, code-mixed Urdu-English  
- **License:** MIT
- **Finetuned from model:** `xlm-roberta-base

### Model Sources

- **Pretrained Base Model:** [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base)

## Uses
### Direct Use

- Classifying sentiment in Roman Urdu social media text.
- Performing automated content moderation.
- Enabling analytics on Roman Urdu datasets.

### Downstream Use

- Can be used as a feature extractor for multi-task NLP pipelines.
- Can be adapted to other Roman Urdu classification tasks with further fine-tuning.

### Out-of-Scope Use

- Not suitable for fully understanding Urdu script (non-Roman Urdu) without transliteration.
- Not intended for detecting sarcasm, irony, or extremely domain-specific slang beyond the training data.

---

## Bias, Risks, and Limitations

- The model may inherit biases from the training data, including slang, offensive terms, and informal language.
- Accuracy may drop for **extremely rare or new Roman Urdu slang** not seen during training.
- May misclassify mixed-script or heavily non-standard text.

### Recommendations

- Users should always validate predictions in critical applications.
- Consider retraining or augmenting with additional domain-specific data for specialized use-cases.

## How to Get Started with the Model

### Quick Start
```python
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="Khubaib01/roman-urdu-sentiment-xlm-r",
    truncation=True
)

pipe("ye banda bohot acha hai")
```

### Advanced Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "Khubaib01/roman-urdu-sentiment-xlm-r"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

inputs = tokenizer("ye insan acha nahi lagta", return_tensors="pt")
logits = model(**inputs).logits
label_id = logits.argmax(dim=-1).item()

id2label = {0:"Positive", 1:"Negative", 2:"Neutral"}
print(id2label[label_id])

```

### Training Details
Training Data

- Sources: WhatsApp chat exports, YouTube comments in Roman Urdu.
- Preprocessing: Removed media messages, deleted messages, emoji-only messages, and very short messages.
- Dataset size: 54k sentences (18k per class, balanced)
- Labels: Positive, Negative, Neutral

### Training Procedure

- Base Model: xlm-roberta-base
- Training regime: fp16 mixed precision, batch size 16, gradient accumulation 4
- Epochs: 4
- Learning Rate: 2e-5
- Optimizer: AdamW
- Evaluation Strategy: Epoch-level evaluation with stratified split

## Evaluation
Evaluation
Testing Data

10% held-out subset of the balanced dataset (stratified per class)

#### Metrics

- Accuracy, F1-score per class
- Confusion matrix to inspect misclassification between classes

#### Results

- Overall accuracy: ~85–90% (depending on dataset variation)
- Accuracy and F1 on vaidation data is 81.57%
- Balanced F1 across Positive, Negative, Neutral


## Environmental Impact

- Hardware: NVIDIA T4 GPU (Colab)
- Training Time: ~1–2 hours
- Precision: fp16
- Estimated CO₂ Emissions: Low (approx. 1–2 kg CO₂eq, Colab)

## Technical Specifications

- Model Architecture: XLM-RoBERTa Base (Transformer encoder, 12 layers, 768 hidden size)
- Objective: Sentiment classification (3-class softmax output)
- Software: Python, PyTorch, HuggingFace Transformers
- Tokenizer: SentencePiece-based multilingual tokenizer

## Citation

- BibTeX:
```BibTeX
@misc{roman_urdu_sentiment2025,
  title={Roman Urdu Sentiment Analysis Model},
  author={Muhammad Khubaib Ahmad},
  year={2025},
  note={HuggingFace Transformers model, fine-tuned from xlm-roberta-base}
}
```

- APA:

```apa
Muhammad Khubaib Ahmad. (2025). Roman Urdu Sentiment Analysis Model. HuggingFace Transformers. Fine-tuned from xlm-roberta-base.
```

## Glossary

- Roman Urdu: Urdu language written using the Latin alphabet.
- Code-mixed text: Sentences containing a mix of Roman Urdu and English words.
- fp16: Half-precision floating-point training for faster GPU usage.

## Model Card Author

Muhammad Khubaib Ahmad

Model Card Contact

[Gmail: muhammadkhubaibahmad854@gmail.com](muhammadkhubaibahmad854@gmail.com)
[HuggingFace](https://huggingface.co/Khubaib01)
[Kaggle](https://www.kaggle.com/muhammadkhubaibahmad)
[LinkedIn](https://www.linkedin.com/in/muhammad-khubaib-ahmad-)
[Portfolio](https://huggingface.co/spaces/Khubaib01/KhubaibAhmad_Portfolio)
