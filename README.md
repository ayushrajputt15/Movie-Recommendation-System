# 🎬 Movie Recommendation System

## Overview
This project implements a **personalized movie recommendation system** using **collaborative filtering**.
The model learns user and movie patterns from historical ratings and recommends movies a user is likely to enjoy.

## Approach
- Built a recommender system **from scratch in NumPy**
- Used collaborative filtering with latent user–movie interactions
- Optimized training using vectorized cost function and gradient descent
- Applied L2 regularization to prevent overfitting

## Evaluation
- Performed a train/validation split on ratings data
- Achieved a **validation RMSE of ~0.98**, indicating good generalization

## Frontend
- Developed an interactive **Streamlit web application**
- Users can select a User ID and view **Top-10 personalized movie recommendations**

## Dataset
- MovieLens dataset
- The dataset included in this repository is a **reduced sample** for demonstration and GitHub size limits
- Original dataset source: https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system

## Tech Stack
- Python
- NumPy, Pandas
- Streamlit

## How to Run
```bash
pip install streamlit numpy pandas
streamlit run app.py
