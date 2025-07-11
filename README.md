# SMS Spam Detection Engine
A complete Python script for building and training a machine learning model to detect SMS spam. This project handles everything from data acquisition and cleaning to model training, hyperparameter tuning, and deployment.

## ► Project Description
This repository contains the code for a Spam/Ham SMS Classifier. The primary goal is to accurately distinguish legitimate messages (ham) from unsolicited advertisements or malicious messages (spam). The script uses a Multinomial Naive Bayes algorithm, which is highly effective for text classification tasks.

- The entire machine learning pipeline is automated:

1. Fetch Data: The script begins by downloading the raw SMS dataset.

2. Process Text: It then applies a series of natural language processing (NLP) techniques to clean and standardize the text.

3. Train Model: A classification model is trained on the processed data.

4. Optimize & Save: The model's parameters are fine-tuned for best performance, and the final model is saved to disk.

### ► Key Features

* Automated Dataset Handling: Downloads and extracts the UCI SMS Spam Collection dataset automatically.

* Robust NLP Preprocessing:

* Converts text to lowercase.

* Removes irrelevant punctuation and numbers.

* Tokenizes messages into individual words.

* Filters out common stopwords (e.g., "the", "a", "in").

* Stems words to their root form (e.g., "running" -> "run").

Effective Feature Extraction: Uses CountVectorizer with bigrams to capture more context from the text.

Optimized Model: Leverages GridSearchCV to find the best hyperparameters for the Naive Bayes classifier, ensuring high accuracy.

Ready for Deployment: Saves the final, trained model as a spam_detection_model.joblib file and includes a function to POST it to a web server.

### ► How to Use
Prerequisites
Ensure you have Python 3 installed. Then, install the required libraries using pip:

``` pip install pandas requests scikit-learn nltk joblib ```

### Running the Script
Clone this repository or save the script as a .py file.

Open a terminal and navigate to the project directory.

Run the script:

``` python your_script_name.py ```

The script will execute the full pipeline, print its progress at each stage, and save the spam_detection_model.joblib file in the same directory.

► Model Deployment
The final step of the script is designed to upload the trained model to a server.

# This code sends the model to a listening server endpoint
```
url = "URL
model_file_path = "spam_detection_model_path"

with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)
    print("Server Response:")
    print(json.dumps(response.json(), indent=4)) 
```
⚠️ Note: This functionality requires a running server at the specified url that is configured to handle file uploads. If no server is present, this part of the script will raise an error.
