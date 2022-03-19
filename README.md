**Boston House Price Prediction**

This app predicts the Boston House Price Prediction. Data obtained from the scikit-learn datasets . We'll follow these two major steps in our program in this project: 1) Use various machine learning models to predict boston house price in  PredModel.py 
2)predict the Boston House Price using a saved model by taking user input collected using streamlit web app.
Here is the screenshot of the app created using streamlit.

![Screenshot2](https://user-images.githubusercontent.com/83027416/159102726-822d9f4f-ed8a-426e-b6a5-3d78e897d83e.jpg)

#**Model building**
Here we load the data and preprocessed using sklearn preprocessing. Then we use different machine learning models such as LinearRegression,Ridge,PolyRidge , SVR,DecisionTreeRegressor,KNeighborsRegressor for house price prediction. 
By plotting boxplot for score  we identify SVR perfome the best and save that model as pickle file.

#**House price prediction**
house price is predicted using saved SVR model based on user inputs. feature importance is plot.

**Prerequisites**
All the required packages and libraries are listed in file requirements.txt. They can be installed in venv using pip install requirements.txt.
