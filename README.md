# AI-Model-for-Real-Estate-Price-Prediction

# 🏡 Mumbai Real Estate Price Prediction Model

This repository contains the code and dataset for an AI project focused on predicting real estate prices in the Mumbai market. The project utilizes a **Random Forest Regressor** model and is deployed as an interactive web application using **Streamlit**.

## 🚀 Project Overview

The goal is to provide accurate property price estimates based on key features like the number of bedrooms (BHK), total area (Sq. Ft.), and specific suburb location.

| Feature | Details |
| :--- | :--- |
| **Model** | Random Forest Regressor |
| **Target Variable** | Property Price (in Rupees) |
| **Data Source** | `house_price_mumbai.csv` |
| **Deployment** | Streamlit Web Application |

## 📁 Repository Structure

* `app.py`: The main Python script containing the data cleaning, model training pipeline, and the Streamlit UI code.
* `house_price_mumbai.csv`: The raw dataset used for training the model.
* `requirements.txt`: Lists all Python package dependencies.
* `README.md`: This project description file.

## 🛠️ How to Run Locally

Follow these steps to set up and run the interactive price prediction tool on your machine.

### Prerequisites

You must have Python (version 3.7+) installed on your system.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone YOUR_GITHUB_REPO_URL_HERE
    cd Mumbai-House-Price-Prediction
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # For Windows:
    .\venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Launch the Application

Run the Streamlit app from your activated environment:

```bash
streamlit run app.py
