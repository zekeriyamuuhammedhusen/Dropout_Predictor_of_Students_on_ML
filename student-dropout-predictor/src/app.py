import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from data_processing import get_project_root, load_data, preprocess_data
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load the trained model
project_root = get_project_root()
model_dir = os.path.join(project_root, 'model')
model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
model = joblib.load(model_path)

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit app interface
st.markdown("<h1 style='text-align: center;'>Student Dropout Predictor</h1>", unsafe_allow_html=True)

# Add a divider
st.divider()

# Define a placeholder for the message
message_placeholder = st.empty()
message_placeholder.markdown(
    "<h4 style='text-align: center;'>Select parameters and press the predict button to predict the student dropout percentage.</h4>",
    unsafe_allow_html=True
)

# User input for prediction
st.sidebar.subheader('Select parameters for predicting the student dropout percentage')

# School input
school_name = st.sidebar.selectbox('School', options=['GP', 'MS'])

# Gender input
gender_name = st.sidebar.selectbox('Gender', options=['M', 'F'])

# Age input
age = st.sidebar.number_input('Age', min_value=15, max_value=25, value=18)

# Address input
address = st.sidebar.selectbox('Address', options=['Urban', 'Rural'])

# Family size input
family_size = st.sidebar.selectbox('Family Size', options=['Greater than 3', 'Less than or equals to 3'])

# Parental status input
parental_status = st.sidebar.selectbox('Parental Status', options=['Living together', 'Living apart'])

# Mother education input
mother_education = st.sidebar.number_input('Education level of the mother (0 to 4)', min_value=0, max_value=4, value=2)

# Father education input
father_education = st.sidebar.number_input('Education level of the father (0 to 4)', min_value=0, max_value=4, value=2)

# Travel time input
travel_time = st.sidebar.number_input('Time taken to travel to school (in minutes)', min_value=1, max_value=5, value=2)

# Study time input
study_time = st.sidebar.number_input('Weekly study hours', min_value=1, max_value=4, value=2)

# Failure input
failure = st.sidebar.number_input('Number of past class failures', min_value=0, max_value=3, value=1)

# School support input
school_support = st.sidebar.radio('School Support Available?', ['Yes', 'No'])

# Family support input
family_support = st.sidebar.radio('Family Support Available?', ['Yes', 'No'])

# Extra paid class input
extra_class = st.sidebar.radio('Extra Classes Paid?', ['Yes', 'No'])

# Extra curricular activities  input
extra_activities = st.sidebar.radio('Extra Curricular Activities?', ['Yes', 'No'])

# Attended nursery input
attended_nursery = st.sidebar.radio('Attended Nursery?', ['Yes', 'No'])

# Wants higher education input
higher_education = st.sidebar.radio('Wants Higher Education?', ['Yes', 'No'])

# Internet access input
internet_access = st.sidebar.radio('Have Internet Access?', ['Yes', 'No'])

# Relationship input
in_relationship = st.sidebar.radio('In Relationship?', ['Yes', 'No'])

# Family relationship input
family_relationship = st.sidebar.number_input('Quality of family relationships (scale 1 to 5)', min_value=1, max_value=5, value=3)

# Free time input
free_time = st.sidebar.number_input('Amount of free time after school (scale 1 to 5)', min_value=1, max_value=5, value=3)

# Going out input
going_out = st.sidebar.number_input('Frequency of going out with friends (scale 1 to 5)', min_value=1, max_value=5, value=3)

# Weekend alcohol consumption input
weekend_alcohol = st.sidebar.number_input('Alcohol consumption on weekends (scale 1 to 5)', min_value=1, max_value=5, value=3)

# Weekday alcohol consumption input
weekday_alcohol = st.sidebar.number_input('Alcohol consumption on weekdays (scale 1 to 5)', min_value=1, max_value=5, value=3)

# Health status input
health_status = st.sidebar.number_input('Health rating of the student (scale 1 to 5)', min_value=1, max_value=5, value=3)

# Absences input
absence = st.sidebar.number_input('Total number of absences from school', min_value=0, max_value=20, value=3)

# Grade 1 input
grade_1 = st.sidebar.number_input('Grade received in the first assessment (0 - 20)', min_value=0, max_value=20, value=10)

# Grade 2 input
grade_2 = st.sidebar.number_input('Grade received in the second assessment (0 - 20)', min_value=0, max_value=20, value=5)

# Final grade input
final_grade = st.sidebar.number_input('Grade received in the final assessment (0 - 20)', min_value=0, max_value=20, value=15)

# Mother job input (values match dataset)
mother_job = st.sidebar.selectbox("Mother Job", ["at_home", "health", "services", "other", "teacher"])

# Father job input (values match dataset)
father_job = st.sidebar.selectbox("Father Job", ["at_home", "health", "services", "other", "teacher"])

# Reason for choosing school input (values match dataset)
school_reason = st.sidebar.selectbox("Reason for choosing the school", ["course", "home", "reputation", "other"])

# Guardian input (values match dataset)
guardian = st.sidebar.selectbox("Guardian", ["mother", "father", "other"])

# Button to predict dropout
if st.sidebar.button('Predict'):
    # Clear the message
    message_placeholder.empty()

    # Encode inputs
    school_name = 1 if school_name == 'MS' else 0
    gender_name = 1 if gender_name == 'M' else 0
    address = 1 if address == 'Urban' else 0
    family_size = 1 if family_size == 'Greater than 3' else 0
    parental_status = 1 if parental_status == 'Living together' else 0
    school_support = 1 if school_support == 'Yes' else 0
    family_support = 1 if family_support == 'Yes' else 0
    extra_class = 1 if extra_class == 'Yes' else 0
    extra_activities = 1 if extra_activities == 'Yes' else 0
    attended_nursery = 1 if attended_nursery == 'Yes' else 0
    higher_education = 1 if higher_education == 'Yes' else 0
    internet_access = 1 if internet_access == 'Yes' else 0
    in_relationship = 1 if in_relationship == 'Yes' else 0
    mother_job_health = 1 if mother_job == 'health' else 0
    mother_job_other = 1 if mother_job == 'other' else 0
    mother_job_services = 1 if mother_job == 'services' else 0
    mother_job_teacher = 1 if mother_job == 'teacher' else 0
    father_job_health = 1 if father_job == 'health' else 0
    father_job_other = 1 if father_job == 'other' else 0
    father_job_services = 1 if father_job == 'services' else 0
    father_job_teacher = 1 if father_job == 'teacher' else 0
    school_reason_home = 1 if school_reason == 'home' else 0
    school_reason_other = 1 if school_reason == 'other' else 0
    school_reason_reputation = 1 if school_reason == 'reputation' else 0
    guardian_mother = 1 if guardian == 'mother' else 0
    guardian_other = 1 if guardian == 'other' else 0

    # Dataframe for the student input (order should be same as the trained dataframe check X.columns)
    student = pd.DataFrame([{
        'School': school_name,
        'Gender': gender_name,
        'Age': age,
        'Address': address,
        'Family_Size': family_size,
        'Parental_Status': parental_status,
        'Mother_Education': mother_education,
        'Father_Education': father_education,
        'Travel_Time': travel_time,
        'Study_Time': study_time,
        'Number_of_Failures': failure,
        'School_Support': school_support,
        'Family_Support': family_support,
        'Extra_Paid_Class': extra_class,
        'Extra_Curricular_Activities': extra_activities,
        'Attended_Nursery': attended_nursery,
        'Wants_Higher_Education': higher_education,
        'Internet_Access': internet_access,
        'In_Relationship': in_relationship,
        'Family_Relationship': family_relationship,
        'Free_Time': free_time,
        'Going_Out': going_out,
        'Weekend_Alcohol_Consumption': weekend_alcohol,
        'Weekday_Alcohol_Consumption': weekday_alcohol,
        'Health_Status': health_status,
        'Number_of_Absences': absence,
        'Grade_1': grade_1,
        'Grade_2': grade_2,
        'Final_Grade': final_grade,
        'Mother_Job_health': mother_job_health,
        'Mother_Job_other': mother_job_other,
        'Mother_Job_services': mother_job_services,
        'Mother_Job_teacher': mother_job_teacher,
        'Father_Job_health': father_job_health,
        'Father_Job_other': father_job_other,
        'Father_Job_services': father_job_services,
        'Father_Job_teacher': father_job_teacher,
        'Reason_for_Choosing_School_home': school_reason_home,
        'Reason_for_Choosing_School_other': school_reason_other,
        'Reason_for_Choosing_School_reputation': school_reason_reputation,
        'Guardian_mother': guardian_mother,
        'Guardian_other': guardian_other
    }])

    # Predict the dropout probability
    dropout_prob = model.predict_proba(student)[0]

    # Convert probabilities to percentage and round to the nearest whole number
    prob_not_dropping_out = round(dropout_prob[0] * 100)
    prob_dropping_out = round(dropout_prob[1] * 100)

    # Display results
    st.markdown(
        f"<div style='text-align:center;'>Probability of NOT Dropping Out: <strong style='color:green; font-size:20px;'>{prob_not_dropping_out}%</strong></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;'>Probability of Dropping Out: <strong style='color:red; font-size:20px;'>{prob_dropping_out}%</strong></div>",
        unsafe_allow_html=True)

    # Display the final prediction
    if dropout_prob[1] > 0.5:
        st.error("The student is predicted to drop out.")
    else:
        st.success("The student is predicted NOT to drop out.")

# Add a divider
st.divider()

# Loading df for plots
df = load_data()
df = preprocess_data(df)
X = df.drop('Dropped_Out', axis=1)
y = df['Dropped_Out']

# --- Model Performance Visualization ---
st.markdown("<h3 style='text-align: center;'>Model Performance: Confusion Matrix and ROC Curve>", unsafe_allow_html=True)

# Columns for model performance charts
col1, col2 = st.columns(2)  # Create two columns

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Predict on the test data using the pretrained model
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Not Dropout', 'Dropout'])

# Plot the confusion matrix
with col1:
    fig_cm, ax_cm = plt.subplots()
    ax_cm.set_title('Confusion Matrix')
    cmd.plot(ax=ax_cm)
    st.pyplot(fig_cm)

# ROC Curve
with col2:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# Add a divider
st.divider()

# Columns charts
col1, col2 = st.columns(2)  # Create two columns

# Correlation between user selected variables
with col1:
    st.sidebar.header('Select Variables for Correlation Plot')
    columns = df.columns.tolist()
    variable_1 = st.sidebar.selectbox('Select the first variable', columns)
    variable_2 = st.sidebar.selectbox('Select the second variable', columns)
    if variable_1 != variable_2:
        # Plotting the scatter plot with regression line
        st.markdown(f"<h4 style='text-align: center;'>Correlation Between {variable_1} & {variable_2}</h4>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the width and height as needed
        sns.regplot(x=df[variable_1], y=df[variable_2], ax=ax, scatter=False, line_kws={'color': 'red'})
        ax.set_xlabel(variable_1)
        ax.set_ylabel(variable_2)
        st.pyplot(fig)
    else:
        st.write("Please select different variables for the scatter plot.")

# Plot the heatmap
with col2:
    st.markdown(f"<h4 style='text-align: center;'>Correlation Heatmap</h4>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)