# Zomato Restaurant EDA & Rating Prediction
**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn

## Business Problem
What factors drive restaurant ratings on Zomato?
Can we predict a restaurant's rating from its features?

## Key Findings
- Restaurants offering online ordering rated **0.3 points higher** on average
- Table booking availability strongly correlated with ratings above 4.0
- North Indian and Chinese cuisines dominate volume; Continental has highest avg rating
- Approximate cost for two people has a non-linear relationship with ratings
- Location clusters (Koramangala, Indiranagar) show consistently higher ratings

## What I built
- Full EDA pipeline: data cleaning, missing value treatment, outlier detection
- Visualisations: distribution plots, correlation heatmaps, bar charts, scatter plots
- Rating prediction model using Linear Regression (baseline) + feature engineering
- Modular project structure (data/ notebooks/ src/ outputs/)

## Project Structure
```
zomato-analysis/
├── data/           # Raw and cleaned dataset
├── notebooks/      # Jupyter notebooks (EDA + modelling)
├── src/            # Reusable Python scripts
├── outputs/        # Charts and reports
└── requirements.txt
```

## Setup
```bash
git clone https://github.com/Priyanshu-2025/Zomato-Data-Analysis.git
pip install -r requirements.txt
jupyter notebook
```

## Contact
Priyanshu Rawat · priyanshurawat315@gmail.com
LinkedIn: linkedin.com/in/priyanshu-rawat-b63894249
