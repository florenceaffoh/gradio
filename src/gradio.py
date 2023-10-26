import gradio as gr
import pandas as pd
import pickle
import os

# Define a function to load the pickle file
def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

# Load your ML model, transformers, and other necessary components
loaded_components = load_pickle(r'src\assets\ml_components')
model = loaded_components['classification_model']
encoder = loaded_components['cat_preprocessor']

# Define Gradio components for input
input_components = [
    gr.Slider(label="Tenure (months)", minimum=1, maximum=72, step=1),
    gr.Slider(label="Monthly Charges", step=0.05, maximum=7000),
    gr.Slider(label="Total Charges", step=0.05, maximum=10000),
    gr.Radio(label="Senior Citizen", choices=["Yes", "No"]),
    gr.Radio(label="Partner", choices=["Yes", "No"]),
    # Add more components for other input parameters as needed
]

# Define a function to make predictions
def make_prediction(*args):
    input_data = pd.DataFrame({'tenure': [args[0]], 'MonthlyCharges': [args[1]], 'TotalCharges': [args[2]], 
                               'SeniorCitizen': [args[3]], 'Partner': [args[4]]})
    # You may need to add more columns if you have more input parameters

    # Process the data with the transformers
    preprocessed_data = encoder.transform(input_data)
    
    # Make predictions using your model
    prediction = model.predict(preprocessed_data)

    # Convert prediction to human-readable output
    return "Your customer will churn." if prediction[0] == 1 else "Your customer will not churn."

# Create a Gradio interface
iface = gr.Interface(
    fn=make_prediction,
    inputs=input_components,
    outputs=gr.Label("Prediction"),
    live=False  # Set to True if you want live updates as users interact with input components
)

# Define a function to reset input components to default values
def clear_inputs():
    for component in input_components:
        if isinstance(component, gr.Slider):
            component.value = component.minimum
        elif isinstance(component, gr.Radio):
            component.value = component.choices[0]

# Clear the output label
iface.outputs[0].reset()

# Add a "Clear" button to the interface
iface.add_button("Clear", clear_inputs)

# Start the Gradio interface
iface.launch()
