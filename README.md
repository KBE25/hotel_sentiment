<img src="image/image.png">

[Credit: altexsoft.com](https://www.altexsoft.com/blog/sentiment-analysis-hotel-reviews/)

# Hotel Sentiment Analysis

**Author: <a href="https://www.linkedin.com/in/karina-basto-eyzaguirre-203a0445/"> Karina Basto-Eyzaguirre</a>**

## Business Understanding
The modern hospitality industry is driven by customer experience, yet the overwhelming volume of unstructured online reviews makes it impossible for management to manually identify and respond to critical feedback in real-time. This project addresses that challenge by automating the initial screening of guest reviews, freeing up staff and enabling a proactive approach to reputation management. Our model's goal is to rapidly convert a high volume of raw feedback into actionable intelligence, ultimately leading to faster service improvements and enhanced customer loyalty.

## Data Understanding
For this analysis, I used a 26.7K lines dataset of <a href="https://www.kaggle.com/datasets/thedevastator/booking-com-hotel-reviews">hotel reviews from Booking.com</a>
, which includes both textual and structured data. I will use the ranking data to create a sentiment score, making the dataset suitable for a supervised classification model. My hybrid model will be trained on the text reviews as well as contextual numerical and categorical features for a more robust analysis. To ensure an unbiased evaluation, I will split the single file into dedicated training, validation, and testing sets.
                                                                                                
## Data Preparation
Effective data preparation is crucial for developing a robust sentiment analysis model. Our process began by loading a dataset of Booking.com reviews and systematically cleaning it by dropping redundant columns and rows with missing ratings.
Next, we moved into Exploratory Data Analysis (EDA), which comprised of four key parts:
- ***Class Distribution Analysis*** to convert numerical ratings into a binary sentiment score, ultimately selecting a threshold of 9 after analyzing class balance.
- ***Qualitative Text Analysis*** to gain a deep understanding of the raw review text and length distributions, guiding our preprocessing and model design.
- ***Feature Selection*** to identify the most valuable structured features, such as review_month and review_age_days, while removing redundant ones to prevent multicollinearity.
- ***Advanced Text Preprocessing*** to transform raw text by lowercasing, removing noise, converting emojis, and, most importantly, implementing negation handling to preserve accurate sentiment.
- ***Text Feature Extraction (TF-IDF)*** to convert the cleaned and processed text into a numerical vector format, which served as a crucial input for our traditional machine learning models.

## Modelling and Evaluation
This project's modeling phase began with establishing a Tuned Logistic Regression model as a strong, interpretable baseline. This initial model, which was integrated into a robust sklearn pipeline, was trained using TF-IDF vectors for text and a selection of structured features. We then expanded our approach to include advanced models like Support Vector Machines (SVMs) and Gradient Boosting Machines (GBMs). To optimize performance across all models, a rigorous hyperparameter tuning and cross-validation strategy was applied, leveraging RandomizedSearchCV to systematically explore the parameter space and ensure the best-performing, most generalized models were identified and saved.
For Model Evaluation, a comprehensive approach was taken. Beyond standard metrics like accuracy, F1-score, and ROC-AUC, a full confusion matrix was used to deeply understand the model's performance on each class. Crucially, a detailed error analysis was conducted on misclassified samples from our best model. This involved manually reviewing false positives and false negatives to identify and document the model's limitations, particularly its struggle with nuanced or conflicting sentiment—a key weakness of the "bag-of-words" approach. This process was essential for understanding the model's real-world reliability and informing future improvements.

## Recommendations
Based on my findings, I recommend the following for a production-level sentiment analysis system:
1. ***Implement Tuned Logistic Regression:*** Given its superior performance and ease of interpretation, this model is the most recommended choice for providing explainable insights to stakeholders.
2. ***Leverage Hybrid Features:*** The model's use of both text and contextual features (like avg_rating and hotel_name) should be maintained to enhance predictive accuracy.
3. ***Investigate Nuanced Findings:*** The counter-intuitive model findings represent a business opportunity. Further analysis could uncover specific service issues or unmet customer expectations.

#### Limitations of the analysis
Although the model performed well, a detailed error analysis revealed its primary limitation lies in its inability to capture contextual nuances, a common challenge for linear models. My model consistently struggled with mixed-sentiment reviews, often misclassifying:
- False Positives: Negative reviews containing strong positive words.
- False Negatives: Positive reviews mentioning a minor problem.
These patterns highlight the difficulty in capturing complex relationships, sarcasm, and the subjective weighing of information in text.

## For More Information:
See the full analysis in the <a href="https://github.com/KBE25/hotel_sentiment/blob/main/notebook.ipynb">Jupyter Notebook</a>.
The business information can also be found in <a href="">this presentation. </a>

For additional info, contact Karina Basto-Eyzaguirre at karinabastoe@gmail.com.

### Repository Structure
- <a href="https://github.com/KBE25/hotel_sentiment/tree/main/image"> image </a>
- <a href="https://github.com/KBE25/hotel_sentiment/blob/main/.gitignore"> .gitignore </a>
- <a href="https://github.com/KBE25/hotel_sentiment/blob/main/README.md"> README.md </a>
- <a href="https://github.com/KBE25/hotel_sentiment/blob/main/notebook.ipynb"> notebook.ipynb </a>
- <a href=""> presentation </a>
- <a href="https://github.com/KBE25/hotel_sentiment/blob/main/processing_text.py"> helper file: processing_text.py </a>

### Resources
- <a href="https://www.deloitte.com/us/en/Industries/consumer/articles/hotel-guest-experience-strategy.html
">Hotel Guest Experience Strategy</a>
- <a href="https://www.pwc.com/us/en/services/consulting/library/consumer-intelligence-series/future-of-customer-experience.html"> Future of Customer Experience</a>