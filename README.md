# Predicting-Agricultural-Commodity-Prices-An-EDA-and-Machine-Learning-Approach
This project performs an Exploratory Data Analysis (EDA) and builds a machine learning model to predict the Modal Price (most frequent price) of agricultural commodities in India based on market data.
Below is a detailed `README.md` file for your project. It includes sections on project setup, datasets used, tools and technologies implemented, and execution instructions. This README is written in Markdown format, which is commonly used for GitHub repositories or project documentation. You can copy this into a `README.md` file in your project directory and adjust paths or details as needed.

---

# Agricultural Commodity Price Prediction

## Project Overview
This project performs an **Exploratory Data Analysis (EDA)** and builds a **machine learning model** to predict the `Modal Price` (most frequent price) of agricultural commodities in India based on market data. The analysis covers price variability across states, districts, and commodities, while the model leverages a Random Forest Regressor to provide accurate predictions.

**Objective**: Empower farmers, traders, and policymakers with data-driven price forecasts.

**Developed by**: [Your Name]  
**Date**: April 02, 2025  
**Powered by**: xAI's Grok 3

---

## Project Setup

### Prerequisites
- **Python**: Version 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Dependencies**: Install required Python libraries (see Tools & Technologies)

### Installation
1. **Clone the Repository** (if hosted on GitHub):
   ```bash
   git clone https://github.com/yourusername/agri-price-prediction.git
   cd agri-price-prediction
   ```
   Alternatively, download the project files manually.

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install manually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

4. **Prepare the Dataset**:
   - Save your dataset as `market_prices.csv` in the project root directory (or update the file path in the script).
   - See "Datasets Used" for details on preparing the CSV.

---

## Datasets Used

### Source
- **Data Origin**: The dataset is derived from agricultural market price records provided in a comma-separated format.
- **Time Period**: July 27, 2023 - August 2, 2023 (expandable with additional data).
- **Scope**: Covers multiple states in India (e.g., Gujarat, Bihar, Haryana, Kerala).

### Structure
The dataset is stored in a CSV file (`market_prices.csv`) with the following columns:
- **State**: State in India (e.g., Gujarat, Bihar).
- **District**: District within the state (e.g., Amreli, Madhubani).
- **Market**: Specific market within the district (e.g., Damnagar, Jainagar).
- **Commodity**: Type of commodity (e.g., Bhindi, Apple, Tomato).
- **Variety**: Subtype of the commodity (e.g., Bhindi, Local).
- **Grade**: Quality grade (mostly FAQ - Fair Average Quality).
- **Arrival_Date**: Date of arrival at the market (format: DD-MM-YYYY).
- **Min Price**: Minimum price in INR.
- **Max Price**: Maximum price in INR.
- **Modal Price**: Most frequent price in INR (target variable).

### Preparation
1. **Create the CSV**:
   - Copy the dataset from your source (e.g., `<DOCUMENT>` content) into a text editor.
   - Save as `market_prices.csv` with commas separating values and headers as the first row.
   - Example content:
     ```
     State,District,Market,Commodity,Variety,Grade,Arrival_Date,Min Price,Max Price,Modal Price
     Gujarat,Amreli,Damnagar,Bhindi(Ladies Finger),Bhindi,FAQ,27-07-2023,4100,4500,4350
     Gujarat,Amreli,Damnagar,Brinjal,Other,FAQ,27-07-2023,2200,3000,2450
     ...
     ```
2. **Place in Project**: Store `market_prices.csv` in the project root or update the script’s file path.

### Notes
- **Size**: The sample provided has ~10 rows; the full dataset may contain hundreds.
- **Outliers**: Contains extreme values (e.g., 59,000 INR for Bitter Gourd in Bihar), addressed in preprocessing.

---

## Tools & Technologies Implemented

### Programming Language
- **Python**: Version 3.8+ for data analysis and modeling.

### Libraries
- **Pandas**: Data manipulation and loading CSV files.
- **NumPy**: Numerical operations and array handling.
- **Matplotlib**: Basic plotting for visualizations.
- **Seaborn**: Enhanced statistical visualizations (e.g., boxplots, heatmaps).
- **Scikit-learn**: Machine learning tools including `RandomForestRegressor`, `LabelEncoder`, and evaluation metrics.

### Development Environment
- **IDE**: Compatible with Jupyter Notebook, VS Code, PyCharm, or any Python editor.
- **Version Control**: Git (optional, if hosted on GitHub).

### Model
- **Random Forest Regressor**: Chosen for its ability to handle categorical data and non-linear relationships.

---

## Execution Instructions

### Step 1: Verify Setup
- Ensure Python and dependencies are installed.
- Confirm `market_prices.csv` is in the project directory or update the path in the script.

### Step 2: Run the Script
1. **Save the Code**:
   - Copy the Python script (below) into a file named `price_prediction.py` in your project directory.
   - Update the `df = pd.read_csv()` line with your CSV file path (e.g., `'market_prices.csv'` if in the root).

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.preprocessing import LabelEncoder
   from sklearn.metrics import mean_absolute_error, mean
