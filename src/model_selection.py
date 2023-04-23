from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from process_data import *
from models import *

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
    curr_dir = os.getcwd()
    output_path = 'outputs'
    if not os.path.exists(os.path.join(curr_dir, output_path)):
        os.makedirs(os.path.join(curr_dir, output_path))
    output_path = 'outputs'
    models_score_path = output_path + '/models_score.csv'
    models_score_df.to_csv(os.path.join(curr_dir, models_score_path), index=False)

if __name__ == "__main__":
    curr_dir = os.getcwd()
    test_path = 'outputs/test.csv'
    X_test, yTest = read_and_split_data(test_path)
    models_preds_path = 'outputs/models_preds.csv'
    models_preds = pd.read_csv(os.path.join(curr_dir, models_preds_path))
    models_names, models_score = evaluate_models(X_test, yTest, models_preds)
    models_report(models_names, models_score)
