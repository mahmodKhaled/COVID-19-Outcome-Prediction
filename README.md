# COVID-19-Outcome-Prediction
The data used in this project will help to identify whether a person is going to recover from coronavirus symptoms or not based on some pre-defined standard symptoms. These symptoms are based on guidelines given by the _World Health Organization_ (WHO).
The dataset contains 14 major variables that will be having an impact on whether someone has recovered or not, the description of each variable are as follows,
1. _Country:_ where the person resides
2. _Location:_ which part in the Country
3. _Age:_ Classification of the age group for each person, based on WHO Age Group Standard
4. _Gender:_ Male or Female
5. _Visited_Wuhan:_ whether the person has visited Wuhan, China or not
6. _From_Wuhan:_ whether the person is from Wuhan, China or not
7. _Symptoms:_ there are six families of symptoms that are coded in six fields
13. _Time_before_symptoms_appear:_
14. _Result: death (1) or recovered (0)_

> The project is implemented by the programming language **_Python_**

> The project used different **Machine Learning Algorithms**

> The project is written fully in **_Jupyter Notebook_**
## Classifiers Used
- K-Nearest Neighbors
- Logistic Regression
- Naïve Bayes
- Decision Trees
- Support Vector Machines
## Performance Metrics
#### K-Nearest Neighbors

| **Performance Comparisons** | **Class 0** | **Class 1** |
| - | - | - |
| `precision` | 0.96 | 0.83 |
| `recall` | 0.99 | 0.62  | 
| `f1-score` |  0.97  | 0.71 |
#### Logistic Regression

| **Performance Comparisons** | **Class 0** | **Class 1** |
| - | - | - |
| `precision` | 0.96 |  0.71 |
| `recall` | 0.97 | 0.62  | 
| `f1-score` |  0.97  | 0.67 |
#### Naïve Bayes

| **Performance Comparisons** | **Class 0** | **Class 1** |
| - | - | - |
| `precision` | 0.96 |   0.13 |
| `recall` | 0.29 | 0.89  | 
| `f1-score` |  0.45  | 0.22 |
#### Decision Trees

| **Performance Comparisons** | **Class 0** | **Class 1** |
| - | - | - |
| `precision` | 1.00 | 0.89 |
| `recall` | 0.99 | 1.00 | 
| `f1-score` |  0.99  | 0.94 |
#### Support Vector Machines

| **Performance Comparisons** | **Class 0** | **Class 1** |
| - | - | - |
| `precision` | 0.97 | 0.86 |
| `recall` | 0.99 | 0.75  | 
| `f1-score` |  0.98  | 0.80 |
## Project Status
![Build](https://img.shields.io/badge/Build-Finished-brightgreen)
