import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

import sys
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from sqlalchemy.engine import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def load_data(database_filepath):
    """ Loads database into panda dataframe

    Input:
    database_filepath: File path of the database.

    Output:
    X, y = Feature and target variables obtained from the dataframe.
    """

    engine = create_engine(f"sqlite:///{database_filepath}")
    table_name = database_filepath.split("/")[-1][:-3]
    df = pd.read_sql_table(table_name, con=engine)

    X = df.message.values

    # drop unnecessary columns, plus child_alone which only had 0 values
    # as seen during assessment of the dataframe
    y = df.drop(columns=["id", "message", "original", "genre", "child_alone"])

    category_names = list(np.array(y.columns))

    return X, y, category_names


def tokenize(text):
    """ Tokenizer function to process text data during the CountVectorizer
    step.

    Input:
    text: text values from the message column.

    Output:
    clean_tokens: tokenized text, normalized and lemmatized.

    """

    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    words = word_tokenize(text)

    stop_words = stopwords.words("english")
    words = [x for x in words if x not in stop_words]

    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for word in words:
        clean_token = lemmatizer.lemmatize(word).strip().lower()
        cleaned_tokens.append(clean_token)

    return cleaned_tokens


def build_model():
    """ Builds machine learning pipeline taking the values in message column to
    predict the classification under the different categories.

    Input:
    none

    Output:
    model: model using the message values to predict the category of a message.
    """


    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        "tfidf__use_idf": [True, False],
        "clf__estimator__n_estimators": [10, 100],
    }

    model = GridSearchCV(pipeline, parameters, verbose = 4)

    return model

def evaluate_model(model, X_test, y_test, category_names):
    """ Uses the fitted model to predict using the test set. Prints the
    Input:
    model: previously fitted model
    X_test: X values from the test set
    y_test: y values from the test set
    category_names: list of the category names

    Output:
    Prints the classification_report on each of the categories.

    """

    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("category:", category_names[i])
        print(classification_report(y_test.iloc[:, 1], y_pred[:, i]))

def save_model(model, model_filepath):
    """ Saves model into a pickle file.
    Input:
    model: finished machile learning model
    model_filepath: name of the filepath for the model pickle file.

    Output:
    Saved model file.
    """

    with open(model_filepath, "w+") as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
