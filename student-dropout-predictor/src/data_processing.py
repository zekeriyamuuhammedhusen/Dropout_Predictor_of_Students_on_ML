import pandas as pd
import os


def get_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return project_root

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'student dropout.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Use the passed dataframe (do not reload inside this function)
    # Dropping the empty and none values
    df = df.dropna()

    # School binary encoding
    df['School'] = df['School'].map({'GP': 0, 'MS': 1})

    # Gender binary encoding
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

    # Address binary encoding
    df['Address'] = df['Address'].map({'R': 0, 'U': 1}) # R = Rural U = Urban

    # Family size binary coding
    df['Family_Size'] = df['Family_Size'].map({'LE3': 0, 'GT3': 1})  # LE3 = Less than or equals to 3 GT3 = Greater than 3

    # Parental status binary encoding
    df['Parental_Status'] = df['Parental_Status'].map({'T': 0, 'A': 1})  # A for living together, T for living apart

    # Yes or no mapping
    columns_to_encode = ['School_Support', 'Family_Support', 'Extra_Paid_Class', 'Extra_Curricular_Activities', 'Attended_Nursery',
                         'Wants_Higher_Education', 'Internet_Access', 'In_Relationship']
 
    # Map yes/no to 1/0 per column (avoids pandas downcasting FutureWarning)
    for col in columns_to_encode:
        df[col] = df[col].map({'no': 0, 'yes': 1}).astype(int)


    # One hot encode
    df = pd.get_dummies(df, columns=['Mother_Job', 'Father_Job', 'Reason_for_Choosing_School', 'Guardian'], drop_first=True)

    # Print the dataset
    #pd.set_option('display.max_columns', None)
    #print(df.head())

    # Return df
    return df