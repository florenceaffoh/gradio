import gradio as gr
import pandas as pd
import pickle
from PIL import Image
import cv2
import time
import os

# Define a function to load the pickle file
def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)
    

def receiveInputs(tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Partner, Dependents, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
                  DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod):
    """Receive inputs, Process them and make predictions using the ML model
    """
    
    df = pd.DataFrame({'tenure': [tenure],'MonthlyCharges': [MonthlyCharges],'TotalCharges': [TotalCharges],'SeniorCitizen': [SeniorCitizen],'Partner': [Partner],
                      'Dependents': [Dependents],'MultipleLines': [MultipleLines],'InternetService': [InternetService],'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup], 
                      'DeviceProtection': [DeviceProtection],'TechSupport': [TechSupport],'StreamingTV': [StreamingTV],'StreamingMovies': [StreamingMovies],
                      'Contract': [Contract],'PaperlessBilling': [PaperlessBilling],'PaymentMethod': [PaymentMethod]
    })
    
    print(f"Inputs as dataframe: {df.to_markdown()}")



# Load the saved components
loaded_components = load_pickle(os.path.join(r'src\assets\ml_components', 'ml.pkl'))
print(f"\n[Info] ML Components Loaded: {list(loaded_components.keys())}")

# Extracting components from the loaded dictionary
reference_features = loaded_components.get("reference_features")
target = loaded_components.get("target")
transformed_columns = loaded_components.get("transformed_columns")
numerical_columns = loaded_components.get("numerical_columns")
selected_features = loaded_components.get("selected_features")
classification_model = loaded_components.get("classification_model")


# If you want to see the contents of loaded components


df_init =({'tenure': [],'MonthlyCharges': [],'TotalCharges': [],'SeniorCitizen': [],'Partner': [],
                      'Dependents': [],'MultipleLines': [],'InternetService': [],'OnlineSecurity': [], 'OnlineBackup': [], 
                      'DeviceProtection': [],'TechSupport': [],'StreamingTV': [],'StreamingMovies': [],
                      'Contract': [],'PaperlessBilling': [],'PaymentMethod': []})

cols = list(df_init.keys())



num_cols, cat_cols = cols[:4], cols[4:]

print(f"\n[Info] Categorical columns: {','.join(numerical_columns)}")
print(f"\n[Info] Numerical columns: {','.join(cat_cols)}\n")

cat_n_unique = {cat_cols[i]: opt_arrs.tolist()
                for (i, opt_arrs) in enumerate [transformed_columns.categories_]}



cols = list(df_init.keys())
num_cols, cat_cols = cols[:4], cols[4:]

print(f"\n[Info] Categorical columns: {','.join(numerical_columns)}")
print(f"\n[Info] Numerical columns: {','.join(cat_cols)}\n")

cat_n_unique = {cat_cols[i]: opt_arrs.tolist()
                for (i, opt_arrs) in enumerate(transformed_columns.categories_)}

# Interface (Note the change here)
input = ([gr.Dropdown(choices, label=col_name) for col_name, choices in cat_n_unique.items()] +
         [gr.Number(label=col_name) for col_name in num_cols])



def make_prediction(input_data):
  
    # If any preprocessing was done on the data before feeding it to the model during training,
    # you'd replicate that here. This might include scaling, encoding, etc.
    
    # Example: Using the scaler loaded earlier
    # scaled_data = scaler.transform(input_data)
    
    # Make the prediction
    prediction = classification_model.predict(input_data)




demo = gr.Interface(

    input,
    'text',
    examples=[], 
)

if __name__ == "__main__":
    demo.launch(debug=True)







