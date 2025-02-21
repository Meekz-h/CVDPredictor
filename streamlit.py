import streamlit 
import pickle
import pandas as pd

model = pickle.load(open('./production.sav', 'rb'))
scaler = pickle.load(open('./min_max_scaler.sav', 'rb'))

streamlit.title('CVD Predictor')
age = streamlit.number_input('Age')
max_heart_reate = streamlit.number_input('Max Heart Rate:')
exercise_angina = streamlit.text_input('Exercise Pain? [Y|N]')
sex = streamlit.text_input('Sex: [M|F]')



def predict(model, scaler, age, max_heart_reate, sex, exercise_angina):
    example = pd.DataFrame()
    example[['Age', 'MaxHR']] = scaler.transform(pd.DataFrame({'Age': [age], 'MaxHR': [max_heart_reate]}))
    example['Sex'] = [1 if sex == 'M' else 0]
    example['ExerciseAngina'] = [1 if exercise_angina == 'Y' else 0]
    return model.predict_proba(example)[0][1]

streamlit.write(f"You have a {predict(model, scaler, age, max_heart_reate, sex, exercise_angina)*100:.2f} % chance of having Heart Disease")