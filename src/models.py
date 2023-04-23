from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
from process_data import *
import os

def create_model(model_type):
    if model_type == "DecisionTree":
        classifier = DecisionTreeClassifier()
    elif model_type == "KNN":
        initial_n_neighbors_value = random.randint(1 , 30)
        classifier = KNeighborsClassifier(n_neighbors = initial_n_neighbors_value, metric = 'minkowski', p = 2)
    elif model_type == "LogisticRegression":
        classifier = LogisticRegression(random_state=42, solver='liblinear')
    elif model_type == "NaiveBayes":
        classifier = GaussianNB()
    elif model_type == "SVM":
        classifier = SVC(random_state=42)
    else:
        raise Exception("Invalid model type")
    return classifier

def optimize_model(X_train, yTrain, model, model_type):
    if model_type == "LogisticRegression" or model_type == "NaiveBayes":
        best_params = {}
        return best_params
    elif model_type == "DecisionTree":
        param_dist = {
        "criterion" : ["gini" , "entropy"],
        "max_depth" : [2 , 4 , 6 , 8],
        "min_samples_leaf": [1 , 2, 3, 4, 5, 6, 7, 8]
        }
        grid = GridSearchCV(model , param_grid= param_dist , cv= 10 , n_jobs= -1)
    elif model_type == "KNN":
        # define the parameter values that should be searched
        k_range = list(range(1, 31))
        # create a parameter grid: map the parameter names to the values that should be searched
        # simply a python dictionary
        # key: parameter name
        # value: list of values that should be searched for that parameter
        # single key-value pair for param_grid
        param_grid = dict(n_neighbors=k_range)
        # instantiate the grid
        grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
    elif model_type == "SVM":
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf' , 'linear']}
        grid = GridSearchCV(model, param_grid, refit = True, verbose = 3)
    else:
        raise Exception("Invalid model type")
    grid.fit(X_train , yTrain)
    if model_type == "KNN":
        best_params = {'n_neighbors': 7, 'metric': 'minkowski', 'p': 2}
        return best_params
    best_params = grid.best_params_
    return best_params

def train_optimized_model(X_train, yTrain, X_test, best_params, model_type, pred_type):
    if model_type == "DecisionTree":
        model = DecisionTreeClassifier(criterion= best_params["criterion"] , max_depth= best_params["max_depth"] , min_samples_leaf= best_params["min_samples_leaf"])
    elif model_type == "LogisticRegression":
        model = LogisticRegression(random_state=42, solver='liblinear')
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors = best_params['n_neighbors'], metric = best_params['metric'], p = best_params['p'])
    elif model_type == "NaiveBayes":
        model = GaussianNB()
    elif model_type == "SVM":
        model = SVC(C = best_params['C'], gamma = best_params['gamma'], kernel = best_params['kernel'], random_state = 0 , probability=True)
    else:
        raise Exception("Invalid model type")
    model.fit(X_train , yTrain)
    if pred_type == "proba":
        y_pred = model.predict_proba(X_test)[:,1]
    elif pred_type == "class":
        y_pred = model.predict(X_test)
    else:
        raise Exception("Invalid pred type")
    return y_pred

if __name__ == "__main__":
    models = ["DecisionTree" , "KNN" , "LogisticRegression" , "NaiveBayes" , "SVM"]
    curr_dir = os.getcwd()
    train_path = 'C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/outputs/train.csv'
    test_path = 'C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/outputs/test.csv'
    train = pd.read_csv(os.path.join(curr_dir, train_path))
    test = pd.read_csv(os.path.join(curr_dir, test_path))
    X_train, yTrain = split_data(train)
    X_test, yTest = split_data(test)
    models_preds = {}
    for model_type in models:
        model = create_model(model_type)
        best_params = optimize_model(X_train, yTrain, model, model_type)
        y_pred = train_optimized_model(X_train, yTrain, X_test, best_params, model_type, "proba")
        models_preds[model_type] = y_pred
    models_preds_df = pd.DataFrame(models_preds)
    models_preds_df.to_csv('C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/outputs/models_preds.csv', index=False)
    