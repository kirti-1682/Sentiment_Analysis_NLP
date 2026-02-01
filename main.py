"""
Main entry point for the Sentiment Analysis application
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the Flask app
from app.app import app, load_or_train_model

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Sentiment Analysis Application")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('app/templates', exist_ok=True)
    os.makedirs('app/static', exist_ok=True)
    
    # Load or train model
    if load_or_train_model():
        print("\nStarting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load or train model.")
        print("Please make sure 'data/cleaned_data.csv' exists.")
        sys.exit(1)