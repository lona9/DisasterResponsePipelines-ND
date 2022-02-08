import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads the message and categories files and merges them into a single
    dataframe.

    Inputs:
    messages_filepath: filepath of the messages file
    categories_filepath: filepath of the categories files

    Output:
    df - Dataframe obtained from merging the input indivual dataframes together.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id", how="left")
    return df

def clean_data(df):
    """Cleans the string values from the categories column to obtain each
    category name, and separate them into their own columns with their
    corresponding values.

    Input:
    df - Dataframe obtained from the previous loading step.

    Output:
    df - Cleaned dataframe with a column for each category.
    """

    categories = df.categories.str.split(";", expand=True)

    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)


    #drop category columns and replace with new boolean columns
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1)

    #clean rows with invalid values found in assessment
    df = df[df.related != 2]

    #drop duplicate rows
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Saves clean dataframe into a SQLite database.

    Input:
    df: Dataframe obtained from the previous cleaning step.
    database_filename: Name of the database file.

    """
    database_name = database_filename[:-3]
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(database_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
