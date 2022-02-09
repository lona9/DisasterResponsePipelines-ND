# Disaster Response Pipeline Project
-----
### Introduction

This project was created for the Udacity Data Scientist Nanodegree "Disaster Response Pipeline", which covered the essential topics covered during the Data Engineering section. In this project, an ELT Pipeline was built to obtain information from messages sent during disaster events, then a Machile Learning Pipeline to build a model that could classify messages into different categories related to the type of event or request that needs attention. Finally, an app was designed to process user input using the model, and classify the message into categories according to the prediction, and also, to show some visualizations of the data.

-----
### Installations

The following repository requires the following libraries to run properly:
- pandas
- numpy
- sqlalchemy
- plotly
- flask
- nltk
- joblib
- re
- pickle
- sklearn

-----
### Files
- app folder:
  - templates folder: `go.html` and `master.hmtl` files for the app
  - `run.py`: script to initiate the app instance
- data folder:
  - `disaster_categories.csv` and `disaster_messages.csv`: original files with messages and categories information.
  - `process_data.py`: script to run the ETL pipeline
- models folder:
  - `train_classifier.py`: script to run the Machine Learning pipeline

-----
### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. After you've opened the app, you'll see a line for user input which reads "Enter a message to classify":
<img width="700" src="https://user-images.githubusercontent.com/77977899/153095048-7e5ddf6d-68ec-45db-b5ba-258ffe24304e.png">

5. You can type your message in english here and then click on "Classify Message" to get your prediction:
<img width="641" alt="Captura de Pantalla 2022-02-08 a la(s) 20 51 35" src="https://user-images.githubusercontent.com/77977899/153095332-92fae0bc-f236-416c-98be-f855fa77ec7d.png">

6. You can also see some visualizations about the data on the main page of the app.

-----

### Discusion
This project allowed for application of most of the content covered during the Data Engineering lessons, and had different aspects to cover which required for some thinking about the box to face properly. Besides the usual assessments and cleaning steps, which were guided in the classrom, the difference in sample size of the messages for each category would give us different recall and score values, which is later confirmed when the classification report is printed, and can be also noticed when entering messages in the app which are related with categories with higher scores versus the opposite.
Two estimators were tested during the development of this project, RandomForestClassifier and AdaBoostClassifier. The former ended up performing more poorly than the AdaBoostClassifier so the latter was chosen in the end. Some parameters were tested in using GridSearchCV to select the best combination for higher scores.
