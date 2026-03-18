# RLHF Pipeline Evaluation: PPO vs. DPO

## Overview
This repository contains the implementation and evaluation of a Reinforcement Learning from Human Feedback (RLHF) pipeline. The project systematically compares Proximal Policy Optimization (PPO) against Direct Preference Optimization (DPO) using sub-billion parameter Large Language Models (LLMs). It evaluates alignment performance, the "alignment tax" (utility degradation), and algorithmic stability across different architectures and preference datasets.

---

## Repository Structure

* **`Advanced_AIML_Assignment_1_DPO_2x2.ipynb`**: Implements a 2x2 experimental grid comparing DPO alignment across two base architectures (`facebook/opt-125m`, `EleutherAI/pythia-160m`) and two preference datasets (`ultrafeedback`, `Anthropic/hh-rlhf`).
* **`Advanced_AIML_Assignment_1_DPO_Beta_Ablation_OPT_150M.ipynb`**: Conducts a hyperparameter ablation study on the KL-divergence penalty (Beta = 0.1 vs. Beta = 0.5) to observe its effect on reward margin scaling and result quality.
* **`Advanced_AIML_Assignment_1_PPO_Baseline.ipynb`**: Establishes the PPO baseline by training a custom Reward Model from scratch, followed by a full PPO policy optimization loop to evaluate alignment stability and mode collapse.

---

## Prerequisites and Environment Setup

These notebooks are designed to be executed in **Google Colab** with GPU acceleration (A100 recommended due to memory constraints during RLHF).

### 1. Google Drive Integration
The training pipelines utilize sequential memory management and direct-to-drive checkpointing to prevent Out-Of-Memory (OOM) errors and data loss. 
* You must have Google Drive mounted in your Colab environment.
* The notebooks are hardcoded to save models, logs, and evaluation graphs to: `/content/drive/MyDrive/RLHF_Project`. Ensure sufficient storage space before running.

### 2. Weights & Biases (Weave) API
Experiment tracking and run logging are managed via the `weave` library.
* You will need an active [Weights & Biases (W&B)](https://wandb.ai/) account.
* When prompted during notebook execution, paste your W&B API key to authenticate and log the training runs to your dashboard.

### 3. Dependencies
The notebooks automatically manage their own dependencies, uninstalling conflicting libraries and installing exact versions required for the `trl` pipeline. Key libraries include:
* `torch==2.4.1`
* `transformers==4.44.2`
* `trl==0.10.1`
* `peft`
* `datasets`

---

## How to Run

1. Open the desired notebook in Google Colab.
2. Enable a GPU runtime (Runtime > Change runtime type > T4 GPU or A100 GPU).
3. Run the setup cells to install dependencies and authenticate Google Drive.
4. Paste your W&B API key when the `weave.init()` cell prompts you.
5. Execute the remaining cells sequentially. The notebooks will automatically handle Supervised Fine-Tuning (SFT), Reward Modeling/Alignment, and final Perplexity evaluations.
