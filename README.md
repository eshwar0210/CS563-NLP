# Product Review Analysis for Indian Market

## Project Overview
This project collects and analyzes product reviews to predict whether products will be useful in India. The dataset consists of reviews for 50 products, each with 500 reviews, and is specifically focused on feedback from Indian users. The process includes sentiment analysis, aspect-based opinion mining, and the use of classification models to make predictions for future products.

---

## Task 1: Dataset Collection

### Objectives:
- Collect a dataset of 50 products, each with 500 reviews, from Indian users.
- Annotate whether each product is useful in India based on the review text.

### Method:
1. **Data Collection**: Automatically scrape and collect product review data from relevant sources that focus on Indian users' feedback.
2. **Annotation**: Based on the review content, classify whether the product is deemed "useful" or "not useful" for Indian users.

---

## Task 2: Perform Analysis

### Sentiment Analysis:
- **Objective**: Understand the overall sentiment of each product by analyzing whether users like or dislike the product.
- **Approach**: Using sentiment analysis tools to extract sentiment scores (positive, negative, neutral) from user reviews.

### Aspect-Based Opinion Mining:
- **Objective**: Identify specific parts of the product that are either appreciated or criticized.
- **Approach**: 
    - Extract and categorize aspects (e.g., quality, packaging, performance, etc.) mentioned in the reviews.
    - Classify the sentiment related to each aspect (positive/negative/neutral).

### Summary of User Reactions:
- **Objective**: Display a summary of overall user reactions.
- **Approach**: Combine sentiment and aspect-based analysis to create a comprehensive view of user feedback, summarizing what users liked and disliked about each product.

---

## Task 3: Classification Model

### Objective:
- Predict whether future products will be useful in India based on the analysis from the previous tasks.

### Approach:
1. **Feature Extraction**: Use the insights gained from sentiment analysis and aspect-based opinion mining as features.
2. **Model Selection**: Run a classification model ( Random Forest ) to predict the usefulness of a product in India.
3. **Model Evaluation**: Assess model performance using metrics like accuracy, precision, recall, F1 score, and ROC AUC.



### Task 1: Dataset Collection

For Task 1, we focused on collecting and annotating product reviews. The process is divided into two main parts:

1. **Web Scraping Notebook**:
   - This notebook is used to **scrape product reviews** from the Amazon India website. It collects reviews for 50 products, with 500 reviews per product, and saves them in a CSV file named `data.csv`. The data includes details such as `product_id`, `review_rating`, `review_title`, `review_description`, and `user_name`. 
   - The notebook ensures that only reviews from Indian users are collected.

2. **Annotate Notebook** (`annotate.ipynb`):
   - The `annotate.ipynb` notebook performs **sentiment analysis and annotation** on the collected review data to determine whether the product is "useful" or "not useful" based on the reviews.
   - **Sentiment Analysis** is performed using three different sentiment analysis methods:
     - **TextBlob**: This method uses TextBlob’s polarity score, which is scaled from 0 to 1.
     - **VADER**: VADER sentiment analysis is used to compute a compound score for each review, also scaled from 0 to 1.
     - **BERT-based Transformer Model**: A pre-trained DistilBERT model is used for sentiment classification, providing a sentiment score scaled from 0 to 1.
   
   These methods provide a comprehensive sentiment evaluation, and their results are stored in new columns: `textblob`, `vader`, and `bert`. The overall sentiment score is calculated by applying a threshold to determine if a product is "useful" or "not useful" based on the reviews.

   Additionally, a **scaled rating** is computed by scaling the `review_rating` to a 0-1 range for further analysis.

   After applying the sentiment analysis, the distribution of verdicts is as follows:
   Useful 41 Not Useful 9
   
   The final annotated dataset is then saved for further analysis and model training.

### Task 2: Perform Analysis

In Task 2, we performed sentiment analysis and aspect-based opinion mining to gain insights into the reviews. The analysis consists of three primary steps:

1. **Sentiment Analysis per Review**:
   - We applied **VADER sentiment analysis** to each product review to determine the overall sentiment. The sentiment scores were then classified as **Positive**, **Negative**, or **Neutral** based on thresholds:
     - Positive: Sentiment score > 0.05
     - Negative: Sentiment score < -0.05
     - Neutral: Otherwise
   - The sentiment analysis results were stored in two new columns: `sentiment_score` (numeric sentiment score) and `sentiment_label` (sentiment classification).

2. **Aspect-Based Sentiment per Product**:
   - **Aspect-based sentiment analysis** was performed by extracting noun chunks (representing aspects of the product) from each review. Using the **spaCy** NLP model, we lemmatized the noun chunks to ensure consistency.
   - For each review, the sentiment of the aspect was classified as Positive, Negative, or Neutral. The aspect-level sentiment for each product was recorded and categorized under each aspect.

3. **Product-Level Sentiment Summary**:
   - We summarized the sentiment for each product based on the total number of positive, negative, and neutral reviews.
   - A final **overall verdict** was derived for each product: "Liked" if the positive reviews outnumber the negative ones, otherwise "Not Liked".
   - Additionally, we summarized the most appreciated and criticized aspects of each product by counting the frequency of positive and negative mentions for each aspect.

The results of this analysis were saved into two CSV files:
- **`product_overall_sentiment_summary.csv`**: Contains a summary of the overall sentiment and verdict for each product, along with the top appreciated and criticized aspects.
- **`aspect_based_opinion_per_product.csv`**: Contains detailed aspect-based sentiment feedback for each product, including the number of positive and negative mentions for each aspect.

This analysis helps us understand both the general sentiment towards each product as well as specific aspects that users like or dislike.

---
Approach for Task 3: Product Review Classification Model
In this task, the goal is to build a machine learning model that classifies product reviews as either "Useful" or "Not Useful". Below is a step-by-step breakdown of the approach followed in the code:

1. Data Preprocessing and Merging
Loading Data: Three datasets are loaded:

data.csv: Contains individual product reviews.

product_sentiment_analysis_combined.csv: Contains sentiment scores and the verdict (Useful/Not Useful).

product_overall_sentiment_summary.csv: Contains top-appreciated and criticized aspects of each product.

Merging Datasets: The three datasets are merged on the product_id field, creating a comprehensive dataset that includes user ratings, sentiment scores, and product features.

Handling Missing Values: Missing values are filled with an empty string in the non-numeric columns to ensure that the model can process the data without errors.

2. Feature Engineering
User Average Rating: For each user, the average review rating is calculated.

Sentiment Score: The average sentiment score for each product is included.

Textual Features: A combined text feature (full_text) is created by concatenating the review_title, review_description, top_appreciated_aspects, and top_criticized_aspects.

Target Variable: The verdict column (Useful/Not Useful) is mapped to a binary target variable (1 for Useful and 0 for Not Useful).

3. Exploratory Data Analysis (EDA)
Verdict Distribution: A count plot is used to visualize the distribution of "Useful" and "Not Useful" reviews.

Correlation Heatmap: A heatmap is generated to show the correlation between numeric features like review_rating, user_avg_rating, and avg_sentiment_score.

Sentiment Score Distribution: A histogram is plotted to show the distribution of sentiment scores across reviews.

Review Description Length Distribution: A histogram is created to visualize the word count distribution of review descriptions.

4. Model Training and Evaluation
Train-Test Split: The data is split into a training set (60%) and a test set (40%) for model evaluation.

Preprocessing Pipeline:

Text Data: The full_text feature is processed using a TF-IDF Vectorizer to extract relevant features from the review text.

Numeric Data: Features like review_rating, user_avg_rating, and avg_sentiment_score are scaled using StandardScaler.

Model Selection: A Random Forest Classifier is chosen for classification due to its ability to handle both numeric and text features effectively.

Model Training: A Pipeline is created to first preprocess the data and then train the model using the RandomForestClassifier.

Evaluation Metrics:

Accuracy: Proportion of correct predictions.

Precision: Proportion of true positives out of all predicted positives.

Recall: Proportion of true positives out of all actual positives.

F1-Score: Harmonic mean of precision and recall.

ROC AUC: Measures the ability of the model to discriminate between useful and not useful reviews.

5. Model Evaluation
Confusion Matrix: A confusion matrix is generated for each model to visualize the true positives, false positives, true negatives, and false negatives.

ROC Curve: A Receiver Operating Characteristic (ROC) curve is plotted to compare the true positive rate against the false positive rate. The AUC (Area Under Curve) is used as a measure of the model's performance.

Final Model Comparison: All evaluation metrics are aggregated into a results table, and the models are compared based on the F1-Score.

6. Results
Model Evaluation Results: After training, the evaluation results (accuracy, precision, recall, F1-score, ROC AUC) are saved in a CSV file model_evaluation_results.csv.

Training Data: The final training data used for model fitting is saved in a CSV file training_data.csv.


### Plots:
The following plots were generated to help visualize the results:

1. **Distribution of Sentiment Scores**: 
   - This plot shows the distribution of sentiment scores across the product reviews.
   - Helps in understanding the overall sentiment trends across products.

2. **Correlation Matrix for Numerical Features**:
   - A heatmap that displays the correlation between numerical features such as `review_rating`, `user_avg_rating`, and `avg_sentiment_score`.
   - Highlights potential relationships between these features and how they might impact model performance.

3. **Review Length Distribution**:
   - A histogram that illustrates the distribution of the number of words in review descriptions.
   - Helps in understanding the average length of reviews and their possible impact on sentiment analysis.

4. **Confusion Matrix for Model Evaluation**:
   - A confusion matrix that provides insight into the classification performance of the model.
   - Displays the count of true positives, true negatives, false positives, and false negatives.

5. **ROC Curve Comparison Between Models**:
   - A plot that compares the ROC curve for multiple models.
   - The ROC curve helps visualize the trade-off between sensitivity and specificity, and the AUC score is used to evaluate model performance.



## How to Run the Project

### Prerequisites:
Take the notebooks into google colab because the dependices between spacy and numpy are very clumsy disturbing other venv.
Everything is working good in google colab.



## Future Work

- **Improve Data Collection**: 
  - Gather more diverse data from a wider range of products, which will help improve the generalizability of the model.

- **Refine Aspect Mining**: 
  - Enhance the aspect extraction process to cover more detailed aspects of the product. This will improve the granularity of the analysis.

- **Model Tuning**: 
  - Tune the hyperparameters of the model to improve prediction accuracy. This can involve techniques like grid search or random search to find the optimal set of parameters.

- **Deploying the Model**: 
  - Build a web interface where users can input product reviews, and the model can predict whether the product is useful in India. This will make the model more accessible to end-users.

## License

This project is licensed under the MIT License 



