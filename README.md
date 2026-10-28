# AI-Enabled Zero-Trust Network Traffic Monitoring and Anomaly Detection

## Overview
This project implements a network anomaly detection system using machine learning and a simulated Zero-Trust policy. It uses the UNSW-NB15 dataset to classify network traffic as normal or malicious.

## Features
- Data preprocessing and categorical feature encoding
- Random Forest and Logistic Regression classifiers
- Rule-based baseline detection
- Simulated Zero-Trust decisions: ALLOW, FLAG, and DENY
- Evaluation using accuracy, precision, recall, and F1-score
- Experiments with clean, noisy, missing-data, and different attack-ratio scenarios

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dataset
The project uses the UNSW-NB15 training and testing datasets.

Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset


## Installation
Clone the repository:

```bash
git clone https://github.com/shannon-fernandes/zero-trust-anomaly-detection.git
cd zero-trust-anomaly-detection
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python src/main.py
```


## Results
The system compares the performance of the implemented detection methods under multiple test conditions. The results and visualizations are available in the project's results directory, if included in the repository.

On the current clean test data, Random Forest achieved 87.60% accuracy and an F1-score of 89.73%.

## Limitations
The system operates offline using dataset records. Zero-Trust access decisions are simulated and are not enforced on a live network.

## Team
- Rashi Shrivastava
- Shannon Fernandes
- Sanika Sujitkumar Khade
- Yash Agarwal

## Course
Computer Networks - Vellore Institute of Technology, Vellore
