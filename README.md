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

## Benchmark Results

### Slang-heavy dataset
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Khubaib01/roman-urdu-sentiment-xlm-r | 0.8444 | 0.8351 |
| ayeshasameer/xlm-roberta-roman-urdu-sentiment | 0.7833 | 0.7803 |
| tahamueed23/urdu-roman-urdu-sentiment | 0.4333 | 0.3374 |
| Aimlab/xlm-roberta-roman-urdu-finetuned | 0.2833 | 0.2084 |

### Formal/Test dataset
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Khubaib01/roman-urdu-sentiment-xlm-r | 0.7508 | 0.7318 |
| ayeshasameer/xlm-roberta-roman-urdu-sentiment | 0.6246 | 0.6246 |
| tahamueed23/urdu-roman-urdu-sentiment | 0.4850 | 0.3872 |
| Aimlab/xlm-roberta-roman-urdu-finetuned | 0.2558 | 0.2171 |

> Benchmark notebook: [Kaggle](https://www.kaggle.com/code/muhammadkhubaibahmad/romanurdu-sentiment-xlm-r-benchmarking)

## Model Details
### Model Description

This model is a fine-tuned XLM-RoBERTa Base transformer for Roman Urdu sentiment analysis. It was trained on a balanced and diverse dataset of ~99k sentences, with labels: Positive, Negative, and Neutral. The dataset includes messages from WhatsApp chats, YouTube comments, Twitter posts, and other social media sources. Preprocessing removed media-only messages, deleted messages, very short messages, and pure emoji messages.

- Developed by: Muhammad Khubaib Ahmad
- Model type: XLM-RoBERTa Base (transformers)
- Language(s): Roman Urdu, code-mixed Urdu-English
- License: MIT
- Finetuned from: xlm-roberta-base

## Model Sources

Pretrained Base Model: xlm-roberta-base

## Uses
## Direct Use

- Classifying sentiment in Roman Urdu social media text.
- Performing automated content moderation.
- Detecting offensive, abusive, or highly negative content in Roman Urdu text due to training on slang-heavy negative messages.
- Enabling analytics on Roman Urdu datasets, such as understanding user feedback or public opinion.

## Downstream Use

- Can be used as a feature extractor in multi-task NLP pipelines.
- Adaptable to other Roman Urdu classification tasks with additional fine-tuning.

## Out-of-Scope Use

- Not suitable for native Urdu script without transliteration.
- May not reliably detect subtle sarcasm, irony, or context-specific toxicity.
- Performance may drop on unseen slang, newly coined offensive terms, or heavily domain-specific expressions.

## Bias, Risks, and Limitations

- The model may inherit biases from the training data, including slang, offensive terms, and informal language.
- Accuracy may be lower on rare, new, or unseen Roman Urdu slang, especially outside social media contexts.
- Heavily code-mixed or non-standard text may be misclassified.
- Should not be relied on as a dedicated toxicity detection tool; it flags negative/offensive content as part of its sentiment predictions.

## Recommendations

- Validate predictions in critical applications.
- Consider additional fine-tuning for domain-specific use cases.

## How to Get Started with the Model
## Quick Start
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

## Training Details
### Training Data

- Sources: WhatsApp chat exports, YouTube comments, Twitter, and other social media.
- Dataset size: ~99k sentences (balanced across Positive, Negative, Neutral).
- Preprocessing: Removed media-only, deleted, emoji-only, and very short messages.

## Training Procedure

- Base Model: xlm-roberta-base
- Training regime: fp16 mixed precision, batch size 16, gradient accumulation 4
- Epochs: 2
- Learning Rate: 1e-5
- Optimizer: AdamW
- Evaluation: Epoch-level evaluation using stratified train/validation split

## Evaluation
### Testing Data

- 10% held-out subset of the robust 99k dataset (stratified per class)

## Metrics

- Accuracy and Macro F1 per class
- Confusion matrix for analyzing misclassifications

## Results

- Overall Accuracy: ~84–85%
- Macro F1: ~83–84%
- Balanced performance across Positive, Negative, and Neutral classes
- Benchmarking against other Roman Urdu sentiment models shows superior performance, especially on slang-heavy texts

## Environmental Impact

- Hardware: NVIDIA T4 GPU (Colab)
- Training Time: ~4–5 hours
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
```APA
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
