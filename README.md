# Prompt Tuning vs Fine-Tuning for Multilingual Sentiment Classification

This project investigates the effectiveness of **parameter-efficient prompt-based tuning methods** compared with traditional **full fine-tuning** for sentiment classification in English and Bangla.

The experiments are implemented using **XLM-RoBERTa** and evaluate multiple tuning strategies across English IMDb reviews and Bangla sentiment datasets.

## Project Overview

Large multilingual transformer models perform well on NLP tasks, but full fine-tuning requires updating all model parameters, which can be computationally expensive. This project compares full fine-tuning with several Parameter-Efficient Fine-Tuning methods:

- Full Fine-Tuning
- Soft Prompt Tuning
- Prefix Tuning
- P-Tuning v2

The goal is to analyze performance, data efficiency, parameter efficiency, and cross-lingual transfer capability.

## Key Features

- English sentiment classification using the IMDb dataset
- Bangla sentiment classification using combined Bangla datasets
- Fine-tuning and prompt-tuning comparison
- Data efficiency experiments with different training sample sizes
- Prompt length ablation study
- Cross-lingual transfer experiment from English to Bangla
- Confusion matrix visualization
- Parameter efficiency analysis
- Statistical significance testing
- IEEE-style figures and result tables generation

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Model

The main model used in this project is:

```text
xlm-roberta-base

XLM-RoBERTa is suitable for multilingual NLP tasks and supports both English and Bangla text.

Dataset

This project uses:

English Dataset
IMDb Movie Review Dataset
Binary sentiment classification:
Negative
Positive
Bangla Dataset

The Bangla dataset is prepared by combining Bangla sentiment datasets such as:

SentNoB
BLP-2023 Bangla Sentiment Dataset

Bangla sentiment labels are handled as:

Negative
Neutral
Positive

For cross-lingual transfer experiments, Bangla labels are converted into binary sentiment format where necessary.

Experimental Methods
1. Full Fine-Tuning

The entire transformer model is updated during training.

2. Soft Prompt Tuning

Only virtual prompt embeddings are trained while the base model remains mostly frozen.

3. Prefix Tuning

Trainable prefix vectors are added to transformer layers to guide model behavior.

4. P-Tuning v2

A deeper prompt-based tuning method that improves parameter-efficient adaptation.

Experiments

The notebook includes the following major experiments:

Main Experiment

Compares all four methods on English and Bangla sentiment classification.

Data Efficiency Experiment

Evaluates model performance using different training sample sizes:

100, 500, 1000, 5000
Prompt Length Ablation

Tests different numbers of virtual prompt tokens:

10, 20, 50, 100
Cross-Lingual Transfer

Trains on English sentiment data and evaluates zero-shot performance on Bangla sentiment data.

Parameter Efficiency

Compares performance with the number of trainable parameters for each method.

Statistical Testing

Uses:

McNemar’s test
Bootstrap confidence intervals

to compare model performance statistically.

Project Structure
.
├── main.ipynb
├── results/
│   ├── data_efficiency_results.csv
│   ├── table2_main_results.csv
│   ├── table2_main_results.xlsx
│   ├── fig5_confusion_matrices.png
│   └── other generated figures
├── saved_models/
│   └── trained model checkpoints
└── README.md
Installation

Create and activate a virtual environment:

python -m venv nlp_project1

For Windows PowerShell:

.\nlp_project1\Scripts\Activate.ps1

Upgrade pip:

python -m pip install --upgrade pip setuptools wheel

Install PyTorch with CUDA support:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install required libraries:

pip install numpy==1.26.4 pandas==2.2.2
pip install transformers==4.40.0 peft==0.10.0 datasets==2.19.0 accelerate==0.29.3
pip install scikit-learn matplotlib seaborn tqdm evaluate openpyxl scipy sentencepiece jupyterlab ipykernel

Add the environment to Jupyter:

python -m ipykernel install --user --name nlp_project1 --display-name "Python (nlp_project1)"
Hardware Configuration

The notebook is configured for local GPU training, especially:

NVIDIA RTX 2060 6GB

Important configuration values:

max_len = 128
batch_size = 4
grad_accum = 4
epochs_ft = 5
epochs_pt = 12
model_name = "xlm-roberta-base"

These settings can be changed in the configuration cell of the notebook.

How to Run
Clone this repository:
git clone https://github.com/TanvirHasan23/Low-Resource-Multilingual-Sentiment-Analysis10.git
Navigate to the project directory:
cd your-repository-name
Create and activate the virtual environment.
Install all dependencies.
Open the notebook:
jupyter lab
Run the notebook cells sequentially from top to bottom.
Output Files

The notebook generates:

Model evaluation results
CSV result files
Excel result files
IEEE-style plots
Confusion matrices
Parameter efficiency figures
LaTeX-ready tables

Generated files are saved mainly inside:

results/

Model checkpoints are saved inside:

saved_models/
Evaluation Metrics

The following metrics are used:

Accuracy
Macro F1-score
Weighted F1-score
Precision
Recall
Confusion Matrix
Trainable parameter percentage
Training time
Research Focus

This project focuses on answering the following questions:

Can prompt-based tuning achieve performance close to full fine-tuning?
Which PEFT method performs best for multilingual sentiment classification?
How effective are prompt-tuning methods in low-resource settings?
How well does English-trained sentiment knowledge transfer to Bangla?
How many trainable parameters are required for competitive performance?
Notes

Before running the Bangla experiments, make sure the Bangla dataset file paths are correctly updated in the notebook.

Example paths used in the notebook:

SENTNOB_TRAIN = "path/to/Train.csv"
SENTNOB_DEV = "path/to/Val.csv"
SENTNOB_TEST = "path/to/Test.csv"

BLP_TRAIN = "path/to/blp23_sentiment_train.csv"
BLP_DEV = "path/to/blp23_sentiment_dev.csv"
BLP_TEST = "path/to/blp23_sentiment_dev_test.csv"

Update these paths according to your local machine or project folder.

Future Improvements
Add command-line training scripts
Add requirements.txt
Add automatic dataset download support
Add support for more Bangla datasets
Add experiment tracking using Weights & Biases or TensorBoard
Deploy trained model using Streamlit or Gradio