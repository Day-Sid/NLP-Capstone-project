"# NLP-Capstone-project" 
"""
A python program that performs sentimet analysis based upon product reviews from a public Amazon customer reviews dataset from Kaggle.

The script does as follows:
- Implements a sentiment analysis model using SpaCy.
- Preprocesses the review text data from the dataset, cleaning the data by removing missing values, stopwords and pucntuation, making each token lowercase, and leminising each token.
- Utilises a modular function to receive a product review and predict its sentiment
- Creates and tests a machine learning linear regression model to predict the sentiment on further reviews with unknown sentiment.
"""

"""
Final report on the project:
1. A description of the dataset used.
The dataset is a large list of consumer reviews for most of Amazon’s owned products. The dataset contains general information about each review, such as; the text, its rating, the product being reviewed, URLs to related pictures, and more. There are 34,000 reviews in total in this dataset
2. Details of the preprocessing steps. 
To preprocess, I first parsed each review individually with spaCY to separate each item in the review as tokenised objects. I then made the reviews entirely lowercase, then went through each token in each review and took out all stop-words and punctuation before each token undergoes lemmatization. This ideally cleans each review to make it easier to predict its sentiment.
3. Evaluation of results.
The resulting model is somewhat accurate, correctly guessing the particularly positive reviews. It has a fairly reasonable Mean Absolute Error of around 0.15 when running with a selected data size of 1500 rows with a scale of -1 to 1. The sentiment scoring ranges are approximately 0.666 in size so some reviews may be placed in the adjacent ranges unintentionally. When all 28,000 useable rows are input into the model, the MAE is 0.07.
4. Insights into the model's strengths and limitations.
The model seems better at identifying positive reviews than negative ones. Negative reviews are not often labelled as negative, by textblob sentiment analysis and therefore by my model. This could be due to the cleaning procedure as it does not handle negations, a complex topic in sentiment analysis. Lemmatization and removal of stop-words could potentially nullify certain phrases such as “not good” to “good” or “useless” to “use”, thereby reducing the overall accuracy of the model.
The model can be modified into increasing the number of sentiment scoring segments to perhaps provide more information as to how strong the sentiment of the review is. The model is limited by use of the textblob sentiment scoring as when comparing the scores to the review itself and the review’s rating, they are often not accurate. Using a more accurate initial sentiment analysis algorithm could help my model’s accuracy by being training on more reliable data.
"""
