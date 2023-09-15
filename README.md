# Sentimental-analysis
Advanced Sentiment Analysis
# Advanced Sentiment Analysis on Tweets

This project aims to perform advanced sentiment analysis on tweets from May 2009. The analysis includes data filtering, enhanced data visualization, advanced data preprocessing, hyperparameter tuning, and model evaluation.

## Project Structure

1. **Data Filtering**: Filter the dataset to include only tweets from May 2009.
2. **Enhanced Data Visualization**: Visualize the distribution of sentiments and the time-series plot of tweets over May 2009.
3. **Advanced Data Preprocessing**: Perform text cleaning and feature extraction using TF-IDF with additional parameters.
4. **Advanced Model Training**: Train machine learning models with hyperparameter tuning using GridSearchCV.
5. **Model Evaluation**: Evaluate the models using accuracy, precision, recall, and F1-score.

## Libraries Used

- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Models Used

1. Optimized Logistic Regression
2. Optimized Bernoulli Naive Bayes
3. Simple Neural Networks

## Performance Metrics

The performance of the models is measured using accuracy, precision, recall, and F1-score for both negative and positive sentiments.

### Optimized Logistic Regression

- **Accuracy**: 74.01%
- **Precision**: 67% (Negative), 78% (Positive)
- **Recall**: 60% (Negative), 82% (Positive)
- **F1-Score**: 63% (Negative), 80% (Positive)

### Optimized Bernoulli Naive Bayes

- **Accuracy**: 74.95%
- **Precision**: 69% (Negative), 78% (Positive)
- **Recall**: 59% (Negative), 84% (Positive)
- **F1-Score**: 64% (Negative), 81% (Positive)

### Simple Neural Networks

- **Accuracy**: 75.47%
- **Precision**: 74% (Negative), 76% (Positive)
- **Recall**: 53% (Negative), 89% (Positive)
- **F1-Score**: 62% (Negative), 82% (Positive)

## Usage

1. Load the dataset and filter it to include only tweets from May 2009.
2. Run the data visualization scripts to understand the distribution of data.
3. Run the data preprocessing scripts to prepare the data for machine learning models.
4. Run the model training and hyperparameter tuning scripts.
5. Evaluate the models and interpret the results.
6. """ 
