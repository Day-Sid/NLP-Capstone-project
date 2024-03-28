'''
T21 - Capstone Project - NLP Applications
'''
import pandas as pd
import spacy

# For sentiment encoding
from textblob import TextBlob

# sklearn used for Vectorising, train_test_split model set up,\
# model fitting, and model accuracy evaluation respectively
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

nlp = spacy.load("en_core_web_md")


def clean_reviews(review):
    '''
    Cleans reviews per review when applied to a rerview dataframe
    Cleaning performed: Lowercase, lemminised, no stopwords, no punctuation
    '''
    # Parses the inputted review and lowercases it
    doc = nlp(review.lower())
    # List created of each cleaned word in the review, unecessary words not included
    cleaned_nlp_review = [token.lemma_ for token in doc
                           if not token.is_stop and not token.is_punct]
    # List joined to reconstruct the review back into a string format
    return " ".join(cleaned_nlp_review)

def sentiment_analysis(review):
    '''
    Cleans an input review and returns its sentiment score as positive, neutral, or negative
    parameter: input review
    returns: predicted sentiment of review
    '''
    cleaned_review = clean_reviews(review)
    vectorized_review = vectorizer.transform([cleaned_review])
    predicted_sentiment = model.predict(vectorized_review)

    # Classifies the floating point of the predicted sentiment into three relevent catagories
    catagorised_rating = ("Positive" if predicted_sentiment > 0.333 else
                            "Negative" if predicted_sentiment < -0.333 else "Neutral")

    return f"The predicted sentiment for this review is: {str(catagorised_rating)}."

# Reading the .csv. Please change the pathway to the appropriate directory if required
reviews = pd.read_csv("C:/Users/sidne/OneDrive/Documents/0 school work/Cogrammar_Course/T21 - Capstone Project - NLP Applications/abridged_amazon_product_reviews.csv")

# Remove all N/A values in the reviews'text column
reviews.dropna(subset = "reviews.text", inplace = True)

# Currently sampling 1500 data rows of the 28000 to reduce CPU usage while maintaining accuracy
cleaned_reviews = reviews.copy().sample(1500,random_state=11)
cleaned_reviews.loc[:,"reviews.text.cleaned"] = cleaned_reviews["reviews.text"].apply(clean_reviews)


# vectorise review text column
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(cleaned_reviews["reviews.text.cleaned"])

# Creating sentiment score for each review
cleaned_reviews.loc[:,"reviews.sentiment.score"] = list(TextBlob(review).sentiment.polarity
                                                for review in cleaned_reviews["reviews.text.cleaned"])

y = cleaned_reviews["reviews.sentiment.score"]

# Training model

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=0.2, random_state=10)

model = LinearRegression()
model.fit(X_train, y_train)

predicted_y = model.predict(X_test)

# Using Mean Absolute Error to assess the model's accuracy
mae = metrics.mean_absolute_error(y_test, predicted_y)
print("The Mean Absolute Error of the model on a scale of -1 to 1: ", mae)


# Takes a review as input, outputs a predicted sentiment score
print(sentiment_analysis(input("Please enter a review to assess its sentiment: ")))

# Testing model on 15 sample reviews and comparing their review rating to their predicted sentiment
print("\n Testing model on a sample set of 15 reviews: \n")
for index, row in cleaned_reviews[["reviews.text","reviews.rating"]].sample(15).iterrows():
    print("Review: ", row["reviews.text"],
          "\nRating /5: ", row["reviews.rating"],
          "\nPredicted sentiment: ", sentiment_analysis(row["reviews.text"]), "\n")
