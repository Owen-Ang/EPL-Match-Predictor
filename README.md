
# English Premier League (EPL) Match Predictor ⚽📊




## Introduction
Welcome to the EPL Match Predictor, a comprehensive machine learning pipeline designed to forecast English Premier League football match outcomes.

Purpose: The goal of this project is to accurately predict two main betting markets: the 1X2 Match Result (Home Win, Draw, Away Win) and the Over/Under 2.5 Goals market. Instead of relying on raw historical data, this project engineers dynamic team form using Exponentially Weighted Moving Averages (EWMA) and integrates real-time team strength using historical ClubElo ratings. The project evaluates models using advanced probabilistic metrics like the Ranked Probability Score (RPS) and Brier Score.

**Target Audience:**

This project is built for data scientists, sports analytics enthusiasts, and algorithmic bettors who want to explore advanced feature engineering, class imbalance handling (SMOTE), model calibration, and ensemble techniques (Random Forest + XGBoost) within the domain of sports forecasting.
##  Table Of Contents
**1. [Introduction](#introduction)**

**2. [Prerequisites & Installation Instructions](#prerequisites--installation-instructions)**

**3. [Methodology & Models](#methodology--models)**

**4. [Evaluation & Insights](#evaluation--insights)**

**5. [Limitations & Future Work](#limitations--future-work)**

**6. [License](#license)**





## Prerequisites & Installation Instructions
Follow these steps to get the project up and running on your local machine. Clarity is key, so make sure you don't skip the data download step! ✨

**1. Prerequisites**
* **Python**: Version 3.8 or higher.
* **Git**: To clone the repository.
* **Jupyter**: To run the .ipynb notebook.

**2. Step-by-Step Installation**

Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/epl-match-predictor.git
cd epl-match-predictor
```

Step 2: Set up a virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Step 3: Install required dependencies
You can install the required packages using pip.
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn scipy jupyter
```

Step 4: Download the Data

Step 5: Launch Jupyter Notebook
```bash
jupyter notebook EPL_Prediction.ipynb
```
## Methodology & Models
* **Feature Engineering (EWMA)**: Teams are dynamically rated based on their recent form. A TimeSeriesSplit cross-validation sweep confirmed an optimal EWMA decay rate of alpha = 0.10 (roughly a 19-game memory window) for tracking stable team quality.  

* **Elo Dominance**: Exploratory Data Analysis revealed Elo Score Difference as the model's anchor feature, showing a clear monotonic pattern with match outcomes (e.g., away wins clustering at a strongly negative Elo differential). Goal-based form metrics (e.g., Net_Goal_Dominance) carried limited linear signal alone, requiring non-linear interactions.  

* **Models Deployed**: 
    * Logistic Regression (baseline for Over/Under 2.5)
    * Feedforward Neural Network (FNN)
    * Soft Voting Ensemble (Random Forest + XGBoost)

* **Imbalance Handling**: SVM-SMOTE was used within the training pipeline to prevent classifiers from defaulting to the majority home-win class. 
## Evaluation & Insights
* **Discrimination**: Both the FNN and the Soft Voting Ensemble achieve comparable and strong discrimination for Away Wins (AUC = 0.71) and Home Wins (AUC = 0.70). However, Draw prediction remains near-random for both architectures (AUC = 0.54), reflecting the highly uncertain margins of tied matches.  

* **Calibration**: The models are reasonably calibrated for Home and Away wins, but draw probabilities suffer from instability, with the FNN showing overconfidence around a 0.60 predicted probability mark.  

* **Statistical Significance**: A paired t-test on the Ranked Probability Scores (RPS) yielded a t-statistic of -2.2144 and a p-value of 0.0271. This statistically significant result proves that the Feedforward Neural Network (mean RPS = 0.2046) outperforms the Ensemble model (mean RPS = 0.2075) on this dataset.  
## Limitations & Future Work
* **Contextual Blindspots**: The current EWMA approach solely captures recent numerical form and fails to account for fundamental team changes such as new managerial appointments or key player transfers.  

* **Draw Prediction**: Predicting draws remains the most challenging aspect due to class imbalance and narrow margins.  

* **Future Iterations**: Future updates will aim to integrate player-level statistics and expected goals (xG). Additionally, advanced temporal models like LSTM networks may be utilized to better capture team synergy and sequential patterns over time. 
## License
This project is licensed under the MIT License - see the [LICENSE](https://choosealicense.com/licenses/mit/)
 file for details.

