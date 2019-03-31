# HeartDisease Prediction
Analyzing various attributes on which Heart Disease directly depends and predicting if a person with a given health conditions has heart disease.

## Motivation
Project was created to make a comparison between the accuracy of predictions obtained from Support Vector Machine and Naivebayes algorithm. The goal of the project is to tune the two algorithms in such a way that the accuracy of prediction is maximized.

## Data-set
Data-set being used in this project can be found on kaggle or you can use this [link](https://www.kaggle.com/ronitf/heart-disease-uci) to go directly to the data-set.

## Screen shots
-HeatMap

![alt text](https://github.com/arnavgarg123/HeartDisease-NaiveBayes-SVM/blob/master/Images/Heatmap.png)

-Output using NaiveBayes<br />  

| **index**  | **precision**  | **recall**  | **f1-score**  | **support**  |  
| ---  | ---  | ---  | ---  | ---  |  
| `0` |      0.71 |     0.57 |     0.64   |     61 |  
|  `1`    |   0.78    |  0.87  |    0.82    |   108 |  
|  **micro avg**    |   0.76   |   0.76    |  0.76    |   169 |  
|  **macro avg**  |     0.75   |   0.72  |    0.73  |     169 |  
| **weighted avg**     |  0.76  |    0.76 |     0.76    |   169 |  

-Output using SVM

| **index** | **precision** |   **recall** | **f1-score** |  **support** |
| --- | --- | --- | --- | --- |
| `0` |      0.73 |     0.61 |     0.66 |       61 |  
| `1`     |  0.80   |   0.87  |    0.83    |   108 |  
| **micro avg**   |    0.78  |    0.78   |   0.78  |     169 |  
| **macro avg**     |  0.76    |  0.74     | 0.75   |    169 |  
| **weighted avg**     |  0.77  |    0.78    |  0.77    |   169 |  

## How to use?
### Clone
- Clone this repo to your local machine using https://github.com/arnavgarg123/HeartDisease-NaiveBayes-SVM.git
### Setup
- Make surer you have jupyter notebook installed on your system with python 3 kernel.
- Using terminal/cmd navigate to the folder containing the files of this repo and run the command `juputer-notebook`.
- Now open NaiveBayes-HeartDisease.ipynb for NaiveBayes and SVM-HeartDisease.ipynb for SVM on jupyter notebook.
 
## Contributing
### Step 1
 - Clone this repo to your local machine using https://github.com/arnavgarg123/HeartDisease-NaiveBayes-SVM.git <br />
### Step 2
 - HACK AWAY! <br />
### Step 3
 - Create a new pull request <br />

## License

[![License](https://img.shields.io/github/license/arnavgarg123/Bangladesh-Rainfall.svg?color=ye)](http://badges.mit-license.org)<br />
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/arnavgarg123/HeartDisease-NaiveBayes-SVM/blob/master/LICENSE.md) file for details
