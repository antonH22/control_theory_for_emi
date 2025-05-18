# Evaluation of a Control Theoretic Approach to Inform Ecological Momentary Interventions

In my thesis, *Using Network Control Theory on Psychological Time Series to Evaluate and Predict the Effects of Smartphone-Based Interventions*, I explore EMA-supported personalized EMI delivery. I focused on one approach using network control theory (NCT) to guide intervention strategies ([paper](https://doi.org/10.1002/mpr.70001), [original GitHub repository](https://github.com/JMFechtelpeter/control_theory_for_emi)).

This repository contains the main codebase used for the analyses. It supports reproduction of tables, figures, and statistical tests **provided you have access to the original dataset, the trained RNN models, and the rnn_module code**, which are not included here due to privacy restrictions.

---

## 1. Usage Guide

### 1.1 Setting Up the Environment  
All required packages are listed in `requirements.txt`. A dedicated virtual environment is recommended.

### 1.2 Folder/File Overview

- **`enhanced_eval_ctrl_approach/`** – Code developed as part of this thesis.  
  - `eval_scripts/`: Contains scripts for computing evaluation results (as described in the Methods section)  
  - `plotting/`: Contains scripts for generating figures used in the thesis  
  - `myutils.py`: Utility functions used in multiple scripts in `eval_scripts/`

- **`ctrl/`** – Sourced from Janik Fechtelpeter’s repository: https://github.com/JMFechtelpeter/control_theory_for_emi 
  - Contains all control theoretic methods and control strategies.

- **`rnn_module/`** – Contains code from a private repository by Janik Fechtelpeter for applying trained RNN models.  
  - To respect the privacy of the original repository, the source files are **not included** here.

- **`data/`** – Placeholder directory.  
  - Place the original dataset here in the required format to run the analyses.

- **`exclude_participant_list.json`** – Specifies dataset files for which no RNN model was trained.