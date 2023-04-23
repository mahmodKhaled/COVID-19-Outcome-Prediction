import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_data():
    current_dir = os.getcwd()
    data_path = 'C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/input/data.csv'
    file_path = os.path.join(current_dir, data_path)
    file = pd.read_csv(file_path, index_col= 0)
    return file

def process_data(file):
    # Dividing dataset into features and result
    x = file[["location" , "country" , "gender" , "age" , "vis_wuhan" , "from_wuhan" , "symptom1" , "symptom2" , "symptom3"
          , "symptom4" , "symptom5" , "symptom6" , "diff_sym_hos"]].values ## features
    y = file.result.values ## result
    # Resampling dataset
    ## splitting the dataset into xTrain , yTrain , xTest , yTest
    xTrain , xTest , yTrain , yTest = train_test_split(x,y, test_size= 0.1 , shuffle= True , random_state= 3)
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(xTrain)
    X_test = sc.transform(xTest)
    return X_train , X_test , yTrain , yTest

def produce_data(X,y, type):
    X_df = pd.DataFrame(X, columns=["location" , "country" , "gender" , "age" , "vis_wuhan" , "from_wuhan" , "symptom1" , "symptom2" , "symptom3"
          , "symptom4" , "symptom5" , "symptom6" , "diff_sym_hos"])
    y_df = pd.DataFrame(y, columns=["result"])
    # merge the two dataframes
    df = pd.concat([X_df, y_df], axis=1)
    # save the dataframe to a csv file
    if type == "train":
        df.to_csv('C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/outputs/train.csv', index=False)
    elif type == "test":
        df.to_csv('C:/Users/mahmo/Desktop/COVID-19-Outcome-Prediction/outputs/test.csv', index=False)

def split_data(data):
    x = data[["location" , "country" , "gender" , "age" , "vis_wuhan" , "from_wuhan" , "symptom1" , "symptom2" , "symptom3"
          , "symptom4" , "symptom5" , "symptom6" , "diff_sym_hos"]].values ## features
    y = data.result.values ## result
    return x, y

if __name__ == "__main__":
    file = read_data()
    X_train , X_test , yTrain , yTest = process_data(file)
    produce_data(X_train, yTrain, "train")
    produce_data(X_test, yTest, "test")
