
# Medical-Insurance-Cost-Prediction
This project aims to predict medical insurance charges using machine learning techniques. The prediction model is built using Gradient Boosting Regression, and the user interface is developed using Streamlit.


## Dataset
The dataset used for training the prediction model is obtained from https://www.kaggle.com/datasets/mirichoi0218/insurance. It contains information about individuals, including their age, sex, BMI, number of children, smoking status, and region, along with their corresponding medical insurance charges.
## Model
The prediction model is built using Gradient Boosting Regression, a powerful ensemble learning technique. Gradient Boosting Regression combines multiple weak learners (decision trees) to create a strong predictive model. The model is trained on the dataset to learn the relationships between the input features and the medical insurance charges.
## User Interface
The user interface is developed using Streamlit, a Python library for building interactive web applications. Users can input their information, including age, sex, BMI, number of children, smoking status, and region, through the Streamlit UI. The prediction model then processes the input data and provides an estimate of the medical insurance charges for the user.
## Usage

To run the application locally, clone the repository, install dependencies, and run the Streamlit application:
```javascript
git clone https://github.com/ishapatel185/Medical-Insurance-Cost-Prediction.git
cd medical-insurance-charges-prediction
pip install -r requirements.txt
streamlit run app.py


```

