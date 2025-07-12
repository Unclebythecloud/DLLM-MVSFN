# Exploring  Truth with Dialogue: Dual Large Language Model Interaction Cooperation and Multi-view Semantic Fusion Network for Fake News Detection

## Overview
This project is designed to classify the authenticity of news using a BERT-based model. It integrates news content, analysis information, and comments to make predictions. This README provides instructions on how to reproduce the code's execution.

## Prerequisites
- **Python**: Version 3.6 or higher is recommended.
- **Libraries**: Install the required libraries using the following command:
```bash
pip install torch transformers pandas scikit-learn tqdm
```

## Project Structure
- `config.py`: Contains configuration settings such as file paths and model hyperparameters.
- `data_loader.py`: Responsible for loading and processing data.
- `model.py`: Defines the model architecture.
- `main.py`: The main script for training and evaluating the model.
- `data/`: Directory containing the data files.

## Step-by-Step Reproduction Guide

### 1. Configure File Paths
Open the `config.py` file and replace the placeholder paths with the actual paths to your data files and the pre-trained BERT model:
```python
# 数据集路径
TRAIN_JSON_PATH = r"path/to/gossipcop_train_set.json"
DEV_JSON_PATH = r"path/to/analyze_val_ch.json"
TEST_JSON_PATH = r"path/to/weibo21_test_newsummary.json"

# 预训练BERT模型路径
BERT_PATH = "path/to/chinese_roberta_wwm_ext"
```

### 2. Prepare the Data
Ensure that your data files (`gossipcop_train_set.json`, `analyze_val_ch.json`, and `weibo21_test_newsummary.json`) are in the correct JSON format. Each JSON object should contain the following fields:
- `text`: The news content.
- `analyze`: The analysis information.
- `comments`: A list of comments.
- `label`: The label indicating the authenticity of the news (0 for real, 1 for fake).

### 3. Run the Code
To train and evaluate the model, run the `main.py` script:
```bash
python main.py
```

### 4. Model Training
- The model will be trained for the number of epochs specified in the `config.py` file (`EPOCHS = 6` by default).
- The training loss will be printed after each epoch.
- The model state will be saved in the `model_saved` directory after each epoch.

### 5. Model Evaluation
- After training, the model will be evaluated on the test set for each saved epoch.
- Evaluation metrics such as accuracy, macro F1 score, AUC, F1 score for real news, and F1 score for fake news will be printed in a table.

## Notes
- The code uses a pre-trained BERT model (`chinese_roberta_wwm_ext`). Make sure you have downloaded the model and specified the correct path in the `config.py` file.
- You can adjust the hyperparameters in the `config.py` file, such as the number of epochs (`EPOCHS`) and the batch size (`BATCH_SIZE`), to suit your needs.

By following these steps, you should be able to reproduce the code's execution and train a model for news authenticity classification.
