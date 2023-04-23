import pandas as pd
import os
from models import *
from model_selection import *
from sklearn.metrics import confusion_matrix, classification_report

def choose_best_model(report):
    best_model = report.idxmax(axis=1)
    best_model = best_model[0]
    return best_model

def confusion_mat(yTest, y_pred):
    cm = confusion_matrix(yTest, y_pred) ## confusion matrix
    cr = classification_report(yTest,y_pred) ## classification report
    return cm, cr

def print_info(best_model, best_params, score, cm, cr):
    print('The best model is: ', best_model)
    print('The best model parameters are: ', best_params)
    print('The best model score is: ', score)
    print('The confusion matrix is: \n', cm)
    print('The classification report is: \n', cr)

def get_train_data():
    curr_dir = os.getcwd()
    train_path = 'outputs/train.csv'
    test_path = 'outputs/test.csv'
    X_train, yTrain = read_and_split_data(train_path)
    X_test, yTest = read_and_split_data(test_path)
    report_path = 'outputs/models_score.csv'
    report = pd.read_csv(os.path.join(curr_dir, report_path))
    return X_train, yTrain, X_test, yTest, report

if __name__ == '__main__':
    X_train, yTrain, X_test, yTest, report = get_train_data()
    best_model = choose_best_model(report)
    model = create_model(best_model)
    best_params = optimize_model(X_train, yTrain, model, best_model)
    y_pred = train_optimized_model(X_train, yTrain, X_test, best_params, best_model, "class")
    cm, cr = confusion_mat(yTest, y_pred)
    score = roc_score(yTest, X_test, y_pred)
    print_info(best_model, best_params, score, cm, cr)