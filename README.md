
# Sentiment Analysis - Amazon Fine Food Reviews

## Introduction
This project performs sentiment analysis on the Amazon Fine Food Reviews dataset. The goal is to analyze customer reviews and determine their sentiment, whether positive or negative, to gain insights into customer opinions and improve product offerings.

## Dataset
The dataset consists of 568,454 food reviews from Amazon. Each review includes the following features:
- **Id**: Unique identifier for the review
- **ProductId**: Unique identifier for the product
- **UserId**: Unique identifier for the user
- **ProfileName**: Profile name of the user
- **HelpfulnessNumerator**: Number of users who found the review helpful
- **HelpfulnessDenominator**: Number of users who indicated whether they found the review helpful
- **Score**: Rating of the product (1 to 5 stars)
- **Time**: Timestamp for the review
- **Summary**: Short summary of the review
- **Text**: Full text of the review

## Project Steps
1. **Read in Data and NLTK Basics**:
   - Import necessary libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `nltk`.
   - Load the dataset and perform initial exploration.

2. **Data Cleaning and Preprocessing**:
   - Handle missing values.
   - Convert text to lowercase and remove punctuation, stop words, and other unnecessary characters.
   - Tokenize the text and perform lemmatization/stemming.

3. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of ratings.
   - Analyze the frequency of words in positive and negative reviews.
   - Generate word clouds to visualize common words.

4. **Feature Extraction**:
   - Convert text data into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

5. **Model Building**:
   - Split the data into training and testing sets.
   - Train various machine learning models (e.g., Logistic Regression, Naive Bayes, Support Vector Machines).
   - Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

6. **Model Evaluation and Interpretation**:
   - Compare the performance of different models.
   - Interpret the results and derive insights from the model predictions.

## Results
The best-performing model and its evaluation metrics will be presented here. Additionally, insights derived from the sentiment analysis will be discussed.

## Conclusion
A summary of the findings, the implications of the results, and potential future work or improvements will be provided.

## Requirements
The following libraries are required to run the notebook:
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/chandanneralgi/Sentiment-Analysis---Amazon-Fine-Food-Reviews.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Sentiment-Analysis---Amazon-Fine-Food-Reviews
   ```
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook "Sentiment Analysis - Amazon Fine Food Reviews.ipynb"
   ```

## Acknowledgements
This project utilizes the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews) available on Kaggle.
