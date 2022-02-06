import sys
import pandas as pd
from sqlalchemy.engine import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer

def load_data(database_filepath):
    """ Loads database into panda dataframe

    Input:
    database_filepath: File path of the database.

    Output:
    X, y = Feature and target variables obtained from the dataframe.
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = database_filepath.split("/")[-1][:-3]
    df = pd.read_sql_table(table_name, con=engine)

    X = df.message

    # drop unnecessary columns, plus child_alone which only had 0 values
    # as seen during assessment of the dataframe
    y = df.drop(columns=["id", "message", "original", "genre", "child_alone"])

    return X, y

def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
