1. First, create the Flask applicationapp.py - Flask Sentiment AnalysisCode from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import2. Create the HTML template with animated backgroundindex.html - Animated Sentiment Analysis PageInteractive artifact 3. Create requirements.txtrequirements.txtCode Flask==3.0.0
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
joblib==1.3.24. Create a main.py runner (alternative to app.py)main.py - Application RunnerCode """
Main entry point for the Sentiment Analysis application
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the Flask app📋 Setup Instructions
Here's how to set up and run your application:
Step 1: Create the project structure
bashsentiment_analysis_project/
│
├── data/
│   └── cleaned_data.csv          # Your CSV file goes here
│
├── models/                        # Will be created automatically
│
├── app/
│   ├── app.py                    # Flask application
│   └── templates/
│       └── index.html            # HTML template
│
├── requirements.txt
└── main.py
Step 2: Install dependencies
Open your terminal in the project directory and run:
bashpip install -r requirements.txt
Step 3: Prepare your CSV file
Make sure your cleaned_data.csv is in the data/ folder. The CSV should have at least 2 columns:

Column 1: Text data (reviews, comments, tweets, etc.)
Column 2: Sentiment labels (positive, negative, neutral)

Example CSV format:
csvtext,sentiment
"This product is amazing!",positive
"Terrible experience, would not recommend",negative
"It's okay, nothing special",neutral
Step 4: Run the application
You have two options:
Option 1 - Using app.py directly:
bashcd app
python app.py
Option 2 - Using main.py (recommended):
bashpython main.py
```

### Step 5: Access the application

Open your web browser and go to:
```
http://localhost:5000
You should see a beautiful animated page with yellow and blue gradients!
🎨 Features
✨ Animated gradient background with yellow (#FFD700) and blue (#4169E1) colors
🎯 Real-time sentiment analysis using machine learning
💫 Floating animations for visual appeal
📊 Confidence scores for predictions
🎭 Color-coded results (green for positive, red for negative, yellow for neutral)
⚡ Responsive design that works on all devices
🔧 Troubleshooting
Issue: Model not loading

Make sure cleaned_data.csv exists in the data/ folder
Check that the CSV has the correct format with text and sentiment columns

Issue: Port 5000 already in use

Change the port in app.py: app.run(debug=True, port=5001)

Issue: Module not found

Make sure you installed all requirements: pip install -r requirements.txt

🚀 Quick Start Commands
bash# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Open browser
# Go to http://localhost:5000
The application will automatically train a model if one doesn't exist, or load the existing model from the models/ folder. Enjoy your beautiful sentiment analysis application! 🎉what about jupyter notebook ├── 01_data_cleaning.ipynb
│   ├── 02_eda_and_feature_extraction .ipynb
│   ├── 03_model_training.ipynb1:14 PM2 / 2I'll create comprehensive Jupyter notebooks for your sentiment analysis project. These will work with your data structure.
1. Data Cleaning Notebook01_data_cleaning.ipynbCode # Sentiment Analysis - Data Cleaning
# This notebook cleans and preprocesses the raw data

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Set displa2. EDA and Feature Extraction Notebook02_eda_and_feature_extraction.ipynbCode # Sentiment Analysis - EDA and Feature Extraction
# This notebook performs exploratory data analysis and feature extraction

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fro3. Model Training Notebook03_model_training.ipynbCode # Sentiment Analysis - Model Training
# This notebook trains and evaluates sentiment analysis models

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warni📚 Complete Instructions for Running Jupyter Notebooks
Step 1: Install Jupyter and Required Libraries
bashpip install jupyter notebook
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
```

### Step 2: Project Structure

Make sure your project has this structure:
```
sentiment_analysis_project/
│
├── data/
│   └── cleaned_data.csv          # Your data file
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_and_feature_extraction.ipynb
│   └── 03_model_training.ipynb
│
├── models/                        # Will be created automatically
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── requirements.txt
└── main.py
Step 3: Create the Notebook Files
For each notebook, create a .ipynb file:

Open terminal in the notebooks/ folder
Start Jupyter:

bash   jupyter notebook

In the Jupyter interface that opens in your browser, click "New" → "Python 3"
Copy and paste the code from each artifact into separate notebooks
Save with the appropriate names

OR use this quick method:
Create Python scripts first, then convert them:
bash# Create .py files with the code
# Then convert to notebooks
jupyter nbconvert --to notebook 01_data_cleaning.py
jupyter nbconvert --to notebook 02_eda_and_feature_extraction.py
jupyter nbconvert --to notebook 03_model_training.py
Step 4: Run the Notebooks in Order
Start Jupyter Notebook:
bashcd sentiment_analysis_project/notebooks
jupyter notebook
Run in sequence:

01_data_cleaning.ipynb

Opens your CSV file
Cleans and preprocesses data
Handles missing values and duplicates
Creates cleaned dataset


02_eda_and_feature_extraction.ipynb

Performs exploratory data analysis
Creates visualizations
Extracts TF-IDF and count features
Generates word clouds


03_model_training.ipynb

Trains multiple ML models
Compares model performance
Saves the best model
Creates confusion matrix and reports



Step 5: Running Each Notebook
In Jupyter Notebook interface:

Click on a notebook to open it
Run cells one by one: Shift + Enter
Or run all cells: Cell → Run All

Step 6: After Running Notebooks
Once all notebooks complete, you'll have:

✅ Cleaned data in data/
✅ Trained models in models/
✅ Visualizations in notebooks/
✅ Ready to run Flask app!

Step 7: Run the Flask Application
bash# From project root directory
python main.py

# Or from app directory
cd app
python app.py
Then open: http://localhost:5000
🎯 Quick Start Commands
bash# 1. Install everything
pip install -r requirements.txt
pip install jupyter notebook wordcloud

# 2. Start Jupyter
cd notebooks
jupyter notebook

# 3. Run notebooks in order (in Jupyter interface)
# - 01_data_cleaning.ipynb
# - 02_eda_and_feature_extraction.ipynb
# - 03_model_training.ipynb

# 4. Run Flask app
cd ..
python main.py

# 5. Open browser
# http://localhost:5000
📊 What Each Notebook Does
Notebook 1 - Data Cleaning:

Loads your CSV
Removes duplicates and missing values
Cleans text data
Detects outliers
Saves processed data

Notebook 2 - EDA & Feature Extraction:

Statistical analysis
Correlation heatmaps
Text analysis (word clouds, distributions)
TF-IDF feature extraction
Count vectorization

Notebook 3 - Model Training:

Trains 5 different models
Compares performance
Saves best model
Creates visualizations
Tests with sample predictions

All visualizations and models will be automatically saved! 🎉#   S e n t i m e n t _ A n a l y s i s _ N L P  
 