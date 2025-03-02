# Sentiment Analysis - Fine-Tuning BERT

## Overview
This repository contains the implementation of sentiment analysis using fine-tuned BERT for classifying negative comments on social media. The model aims to detect and mitigate cyberbullying and harmful language online. This work is based on the research paper:

**"Phân loại bình luận bằng mô hình BERT"**  
*Authors: Nguyễn Văn Huy, Trần Văn Hoàng*  
*Khoa Công nghệ thông tin, Đại học Khoa học Huế*



## Methodology
### Models Implemented:
- **Fine-tuned BERT**: A classification model using BERT with an added output layer.
- **LSTM-CNN with BERT Embeddings**: A hybrid approach combining BERT embeddings with LSTM and CNN for feature extraction.
- **LSTM-CNN Baseline**: A traditional deep learning approach without BERT embeddings.

### Dataset:
- Cyberbullying Classification dataset from Kaggle.
- Contains ~47,000 labeled comments categorized into six types: `religion`, `age`, `gender`, `ethnicity`, `not_cyberbullying`, and `other_cyberbullying`.

### Data Preprocessing:
- Removal of URLs, special characters, and unnecessary whitespaces.
- Conversion to lowercase.
- Tokenization and padding using BERT tokenizer.

## Results
| Model               | Precision | Recall | F1-score | Accuracy |
|---------------------|-----------|--------|----------|----------|
| LSTM-CNN           | 0.64      | 0.63   | 0.63     | 0.63     |
| LSTM-CNN + BERT    | 0.85      | 0.84   | 0.84     | 0.84     |
| BERT Fine-tuning   | 0.73      | 0.71   | 0.69     | 0.71     |

- The **LSTM-CNN with BERT embeddings** model outperforms the others, especially in detecting sensitive categories like `age`, `ethnicity`, and `religion`.
- However, challenges remain in classifying `not_cyberbullying` and `other_cyberbullying`, requiring further optimization and additional training data.

## Installation & Usage
### Prerequisites
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- TensorFlow/Keras (for hybrid model)
- Scikit-learn



