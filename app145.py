import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.neighbors import NearestNeighbors

# Logo image path (replace with your image filename)
logo_image = "logo.png"

with st.sidebar:
    st.title('Your Healthcare Portal')
    st.sidebar.image(logo_image)  # Adjust width as needed

# loading the saved models
diabetes_model = pickle.load(open('./diabetes_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Menu', ['Diabetes Prediction', 'Nutrition Regime', 'Others'],
                           icons=['activity', 'heart', 'person'], default_index=0)
# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level', key="glucose")
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', key="bp")
    with col1:
        Insulin = st.text_input('Insulin Level', key="insulin")
    with col2:
        BMI = st.text_input('BMI value')
    with col3:
        Age = st.text_input('Age of the Person')
    
    if st.button('Diabetes Test Result'):
        try:
            # Convert string input to numeric for prediction
            Pregnancies = int(Pregnancies)
            Glucose = float(Glucose)
            BloodPressure = float(BloodPressure)
            Insulin = float(Insulin)
            BMI = float(BMI)
            Age = int(Age)

            # Predicting diabetes
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, Insulin, BMI, Age]])
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
            st.success(diab_diagnosis)

            # Blood pressure analysis
            if BloodPressure >= 90 and BloodPressure < 130:
                st.info(f"Your blood pressure is normal: {BloodPressure} mmHg.")
            elif BloodPressure >= 130 and BloodPressure <= 139:
                st.warning(f"You are at high risk of prehypertension with blood pressure: {BloodPressure} mmHg. It can lead to serious health issues.")
            elif BloodPressure >= 140 and BloodPressure <= 179:
                st.warning(f"With your blood pressure: {BloodPressure} mmHg, you have been diagnosed with level 1 hypertension.")
            elif BloodPressure >= 180:
                st.error(f"Your blood pressure is {BloodPressure} mmHg, indicating level 2 hypertension.")
            else:
                st.info(f"Your blood pressure is {BloodPressure} mmHg, indicating low blood pressure (hypotension).")
            
            # Glucose analysis
            if Glucose <= 70:
                st.error("This is a very low and dangerous glucose level.")
            elif Glucose > 70 and Glucose <= 100:
                st.info("This is a normal glucose level.")
            elif Glucose > 100 and Glucose <= 126:
                st.warning("You tend to have prediabetes.")
            else:
                st.error("Your glucose is in highly alarming level.")

            # Insulin analysis
            if Insulin < 16:
                st.warning("This is a low level of Insulin.")
            elif Insulin >= 16 and Insulin <= 166:
                st.info("This is a normal level of Insulin.")
            else:
                st.warning("Your insulin too high, you may be related to diabetes, metabolic syndrome, or insulin resistance.")
        
        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")
# # Diabetes Prediction Page
# if selected == 'Diabetes Prediction':
#     st.title('Diabetes Prediction using ML')
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         Pregnancies = st.text_input('Number of Pregnancies')
#     with col2:
#         Glucose = st.text_input('Glucose Level', key="glucose")
#     with col3:
#         BloodPressure = st.text_input('Blood Pressure value', key="bp")
#     with col1:
#         SkinThickness = st.text_input('Skin Thickness value')
#     with col2:
#         Insulin = st.text_input('Insulin Level', key="insulin")
#     with col3:
#         BMI = st.text_input('BMI value')
#     with col1:
#         DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
#     with col2:
#         Age = st.text_input('Age of the Person')
    
#     if st.button('Diabetes Test Result'):
#         # Predicting diabetes
#         diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
#         diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
#         st.success(diab_diagnosis)

#         # Convert string input to integer for analysis
#         bp = int(BloodPressure)
#         glucose_level = int(Glucose)
#         insulin_level = int(Insulin)
        
#         # Blood pressure analysis
#         if bp >= 90 and bp < 130:
#             st.info(f"Your blood pressure is normal: {bp} mmHg.")
#         elif bp >= 130 and bp <= 139:
#             st.warning(f"You are at high risk of prehypertension with blood pressure: {bp} mmHg. It can lead to serious health issues.")
#         elif bp >= 140 and bp <= 179:
#             st.warning(f"With your blood pressure: {bp} mmHg, you have been diagnosed with level 1 hypertension.")
#         elif bp >= 180:
#             st.error(f"Your blood pressure is {bp} mmHg, indicating level 2 hypertension.")
#         else:
#             st.info(f"Your blood pressure is {bp} mmHg, indicating low blood pressure (hypotension).")
        
#         # Glucose analysis
#         if glucose_level <= 70:
#             st.error("This is a very low and dangerous glucose level.")
#         elif glucose_level > 70 and glucose_level <= 100:
#             st.info("This is a normal glucose level.")
#         elif glucose_level > 100 and glucose_level <= 126:
#             st.warning("You tend to have prediabetes.")
#         else:
#             st.error("Your glucose is in highly alarming level.")

#         # Insulin analysis
#         if insulin_level < 16:
#             st.warning("This is a low level of Insulin.")
#         elif insulin_level >= 16 and insulin_level <= 166:
#             st.info("This is a normal level of Insulin.")
#         else:
#             st.warning("Your insulin too high, you may be related to diabetes, metabolic syndrome, or insulin resistance.")

# Nutrition Recommendation Page
if (selected == 'Nutrition Regime'):
    st.title('Automatic Diet Recommendation')
    st.subheader('Modify the values and click the Generate button to use')

    age = st.number_input("Age", min_value=0, max_value=100)
    height_cm = st.number_input("Height (cm)", min_value=0, max_value=220)
    weight_kg = st.number_input("Weight (kg)", min_value=0, max_value=150)
    gender = st.radio("Gender", ("Male", "Female"))
    activity_level = st.selectbox("Activity Level", ("Light exercise", "Little/no exercise", "Extra active (very active & physical job)"))
    weight_loss_plan = st.selectbox("Choose your weight loss plan", ("Maintain weight", "Lose weight", "Gain weight"))
    ingredients = st.multiselect("Select Ingredients", options=['Chicken', 'Fish', 'Beef', 'Vegetables', 'Cake', 'Pasta', 'Salad', 'Egg', 'Smoothies', 'Drink', 'Sweet', 'Shirmp', 'Noodle', 'Cereal'])


    if st.button("Generate Diet Recommendations"):
        # Calculate BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        # Determine BMI Category
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif 18.5 <= bmi < 25:
            bmi_category = "Normal Weight"
        elif 25 <= bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        st.subheader("**BMI CALCULATOR**")
        st.write('Body Mass Index(BMI)')
        st.subheader(f'{bmi:.2f}')
        st.subheader(bmi_category)
        st.write('Healthy BMI range: 18.5 to 24.9')
        st.write(' If your BMI is less than 18.5, it falls within the underweight range.',
    'If your BMI is 18.5 to 24.9, it falls within the Healthy Weight range.',
    'If your BMI is 25.0 to 29.9, it falls within the overweight range.',
    'If your BMI is 30.0 or higher, it falls within the obese range.')

        if gender == "Male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:  # Female
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

        # Adjust BMR based on activity level
        if activity_level == "Light exercise":
            bmr *= 1.375
        elif activity_level == "Little/no exercise":
            bmr *= 1.2
        else:  # Extra active
            bmr *= 1.725

        # Adjust BMR based on activity level
        activity_factors = {"Little/no exercise": 1.2, "Light exercise": 1.375, "Extra active (very active & physical job)": 1.725}
        bmr *= activity_factors[activity_level]

        # Adjust BMR based on weight loss plan
        weight_adjustment = {"Maintain weight": 0, "Lose weight": -500, "Gain weight": 500}
        calorie_needs = bmr + weight_adjustment[weight_loss_plan]

        st.subheader("**CALORIES CALCULATOR**")
        st.write('The result show a number of daily calories estimates that can be used as a guildline for how may calories to consume each day to mauntain, lose, or gain weight at a chosen rate.')
        st.subheader(f"Daily Calorie Needs: {calorie_needs:.0f} calories")
        st.success('⌛ Generating recommendations...')

        st.subheader("**Here are your recommendations:**")

        # Recommend foods based on calorie needs and selected ingredients
        data = pd.read_csv("food_data_1.csv")
        data['ingredients'] = data['ingredients'].fillna('').astype(str)
        
        if ingredients:
            data = data[data['ingredients'].apply(lambda x: all(item in x.split(',') for item in ingredients))]

        if data.empty:
            st.warning("No foods match the selected ingredients. Please adjust your selections.")
        else:
            n_neighbors = min(10, len(data))
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(data[['calories']])
            distances, indices = nn.kneighbors([[calorie_needs]])
            
            recommended_foods = data.iloc[indices[0]]
            for index, row in recommended_foods.iterrows():
                
                st.write(f"Food Type: {row['check_type']}, Food name: {row['food_name']}")
                st.write(f"- Calories: {row['calories']}, Description: {row['food_healthlabel']}")
                st.write(f"- Calcium: {row['Calcium']}, Carbs: {row['Carbs']}, Cholesterol: {row['Cholesterol']}, Fat: {row['Fat']}, Zinc: {row['Zinc']}")

# Others Page Handling
if (selected == 'Others'):
    df = pd.read_csv("food_data_1.csv")
    st.write(df)

# Navigation options
nav_bar = st.sidebar
page = nav_bar.selectbox("Navigate", ["About Us", "Services", "Contact Us"])

if page == "About Us":
    st.subheader("About Us")
    # ... include all the other content as it was before

elif page == "Services":
    st.subheader("Services")
    # ... include all the other content as it was before

elif page == "Contact Us":
    st.subheader("Contact Us")
    # Implementation of contact information and form
