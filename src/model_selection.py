from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from process_data import *
def roc_score(yTest, X_test, preds):
    score = roc_auc_score(yTest, preds)
    return score

def evaluate_models(X_test, yTest, preds):
    model_names = preds.columns
    models_scores = []
    for model_name in model_names:
        models_scores.append(roc_score(yTest, X_test, preds[model_name]))
    return model_names, models_scores   

def models_report(models_names, models_score):
    models_score_dic = {}
    for i in range(len(models_names)):
        models_score_dic[models_names[i]] = models_score[i]
    models_score_df = pd.DataFrame(models_score_dic, columns=models_names, index= [0])
    models_score_df.to_csv(os.path.join(curr_dir, 'outputs/models_score.csv'), index=False)

if __name__ == "__main__":
    curr_dir = os.getcwd()
    test_path = 'C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/outputs/test.csv'
    test = pd.read_csv(os.path.join(curr_dir, test_path))
    X_test, yTest = split_data(test)
    models_preds_path = 'C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/outputs/models_preds.csv'
    models_preds = pd.read_csv(os.path.join(curr_dir, models_preds_path))
    models_names, models_score = evaluate_models(X_test, yTest, models_preds)
    models_report(models_names, models_score)
