# Customer Churn Predictor

A Streamlit-based machine learning app that predicts whether a telecom customer is likely to churn. The app trains a `RandomForestClassifier` on a telecom churn dataset, shows model evaluation charts, explores the underlying data, and provides a simple live prediction interface.

Repository: https://github.com/chauhan-03/customer-churn-predictor

## What The App Does

- Loads a telecom churn dataset from a public CSV URL.
- Falls back to a synthetic dataset if the remote dataset is unavailable.
- Encodes categorical columns for model training.
- Splits the data into train and test sets.
- Trains a Random Forest classifier.
- Displays accuracy, ROC-AUC, a confusion matrix, an ROC curve, and feature importance.
- Shows churn distribution and tenure-based exploratory charts.
- Lets a user enter customer values and get a churn probability.

## Tech Stack

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Python Requirement

- Python 3.12 recommended
- Python 3.10+ should also work for this project

## Project Structure

```text
customer-churn-predictor/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
git clone https://github.com/chauhan-03/customer-churn-predictor.git
cd customer-churn-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the virtual environment already exists, just activate it and install the requirements again if needed.

## Run The App

```bash
source .venv/bin/activate
streamlit run app.py
```

Or run it directly from the project virtual environment:

```bash
./.venv/bin/streamlit run app.py
```

By default Streamlit will print a local URL such as `http://localhost:8501`.

## How It Works

### 1. Data Loading

The app attempts to read a telecom churn CSV from GitHub using `pandas.read_csv()`.

If that request fails, it creates a synthetic dataset with fields such as:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `PhoneService`
- `InternetService`
- `Contract`
- `PaymentMethod`
- `Churn`

### 2. Preprocessing

- Object columns are label-encoded.
- The target column, `Churn`, is converted from `Yes` / `No` into numeric labels.
- Remaining columns are coerced to numeric values.
- Rows with invalid values after conversion are dropped.

### 3. Model Training

The model is trained using:

- `train_test_split(test_size=0.2, random_state=42)`
- `RandomForestClassifier(n_estimators=100, random_state=42)`

This means:

- 80% of the data is used for training.
- 20% is used for testing.
- 100 decision trees are used in the forest.
- `random_state=42` makes runs reproducible.

### 4. Evaluation

The app computes and displays:

- Accuracy
- ROC-AUC
- Confusion matrix
- ROC curve
- Top 10 feature importances

### 5. Live Prediction

The UI accepts customer inputs and generates a churn probability with:

- a binary outcome label
- a probability score
- a simple risk message

## App Sections

### Model Performance

Shows:

- headline metrics
- confusion matrix heatmap
- ROC curve
- feature importance chart

### Data Insights

Shows:

- churn class distribution
- tenure distribution grouped by churn outcome
- a raw sample of the dataset

### Live Prediction

Lets the user provide:

- tenure
- monthly charges
- total charges
- senior citizen status
- contract type
- internet service type

The current prediction payload maps only some of these inputs directly into the model input row. The app is functional, but the live prediction form can still be improved so all selected categorical values are encoded consistently with training.

## Dependencies

The project uses the packages listed in [`requirements.txt`](./requirements.txt):

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Notes And Limitations

- The model is trained at app startup and cached with `@st.cache_data`.
- If the remote dataset is unavailable, the synthetic fallback keeps the app usable.
- The app suppresses warnings globally, which keeps the UI cleaner but can hide useful debugging information.
- The imported `classification_report` metric is currently not displayed in the UI.
- The live prediction form currently does not fully encode all categorical selections into the model input.

## Author

Jatin Chauhan  
GitHub: https://github.com/chauhan-03  
LinkedIn: https://www.linkedin.com/in/jatin-chauhan-a07153171/
