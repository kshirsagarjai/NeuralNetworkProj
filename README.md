# 🧠 Neural Networks – Japanese Vowel Speaker Classification

Semester project for the course **Neural Networks (WBAI028-05)** at the **University of Groningen (RUG)**.
This project explores the application of neural networks to classify speakers of Japanese vowels based on acoustic features.

---

## 📘 Project Overview

The goal of this project is to build and evaluate a neural network model capable of identifying individual speakers from different recordings.

Key objectives:
- to be added

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/OnlyFranko/NN-Semester-Project.git
```

### 2. Create and activate the [Anaconda](https://www.anaconda.com/) environment
```bash
conda env create -f environment.yml
conda activate NN
```

---

## 📂 Project Structure
```
├── data/                     # Test and Train data
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
├── environment.yml           # Conda environment definition
├── .pre-commit-config.yaml   # Pre-commit configuration
└── README.md                 # Project documentation
```

## 📊 Data

#TODO: add something about the dataset

## 🪝 Pre-Commit Hooks
Pre-commit hooks automatically run checks before each Git commit to ensure code quality. They enforce Code formatting with Black, PEP8 compliance with Flake8, import sorting with isort, etc.

### How to Use Pre-Commit
- After installing (`pre-commit install`), hooks run automatically on `git commit`.
- To manually run hooks on all files:
  ```bash
  pre-commit run --all-files
  ```
- To skip hooks for a commit:
  ```bash
  git commit --no-verify -m "message"
