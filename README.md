# Patient Discharge Classificationbased on the Hospital Treatment Process:

This is the supplementary repository for the research on "Patient Discharge Classification based on the Hospital Treatment Process".
Submitted to ICPM Conference 2021 (4th International Workshop on Process-Oriented Data Science for Healthcare (PODS4H)).


## Authors:
- Jonas Cremerius
- Maximilian König
- Christian Warmuth
- Prof. Dr. Mathias Weske

## Abstract

Heart failure is one of the leading causes of hospitalization and rehospitalization in American hospitals, leading to high expenditures and increased medical risk for patients. The discharge location has a strong association with the risk of rehospitalization and mortality, which makes determining the most suitable discharge location for a patient a crucial task. So far, work regarding patient discharge classification is limited to the state of the patients at the end of the treatment, including statistical analysis and machine learning. However, the treatment process has not been considered yet. In this contribution, the methods of process outcome prediction are utilized to predict the discharge location for patients with heart failure by incorporating the patient's department visits and measurements during the treatment process. This paper shows that with the help of convolutional neural networks, an accuracy of 77\% can be achieved for the hospital discharge classification of heart failure patients. The model has been trained and evaluated on the MIMIC-IV real-world data set on hospitalizations in the US.

## Setup 

To install the required packages, please prepare an environment with Python 3.7, navigate to the repository directory and execute
````pip3 install -r requirement.txt````.

You can then start Jupyter Lab by typing ```jupyter lab``` in a terminal in order to access the notebooks containing the code.

We used ```theano``` as the backend for training, and a ````tensorflow```` backend is necessary for the execution of the evaluation. This can be changed by adapting the ```~/.keras/keras.json```.

Executing any code to reproduce our results requires access to the MIMIC IV database (instructions can be found [here](https://physionet.org/news/post/352)). After setting up the database locally using postgres, the ```data_generation/input_data_creation``` notebook extracts the relevant data to the ```data``` directory.
