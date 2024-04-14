import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\Prince Kumar Gupta\OneDrive\Documents\project_163 document\preprocess data\preprocessed_data.csv")
df = df.rename(columns={'Unnamed: 0': 'date'})
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Load the best models
best_models = {}

# Load the best models from their saved files
for column in df.columns:
    # Load the model for this column
    model_path = f"C:\\Users\\Prince Kumar Gupta\\OneDrive\\Documents\\project_163 document\\preprocess data\\best_models_1\\{column}.pkl"
    try:
        loaded_model = joblib.load(model_path)
        best_models[column] = loaded_model
    except FileNotFoundError:
        st.write(f"Model file not found for column: {column}")

# Streamlit App
st.title('Time Series Forecasting App')

# Allow user to select the column
selected_column = st.selectbox('Select Column:', df.columns)

# Allow user to input number of points for forecasting
num_future_points = st.slider('Number of Future Points', min_value=1, max_value=30, value=12, step=1, format="%d", help="Drag the slider or use the arrow keys to select the number of future points for forecasting.")

# Forecast future values
if st.button('Forecast'):
    future_predictions = {}
    try:
        # Forecast future values
        future_forecast = best_models[selected_column].forecast(steps=num_future_points)
        future_predictions[selected_column] = future_forecast
    except Exception as e:
        st.write(f"An error occurred while forecasting for column {selected_column}: {e}")

    # Display predictions
    if future_predictions:
        st.write("Future Predictions:")
        # Convert forecasted data to DataFrame
        forecast_df = pd.DataFrame(future_predictions)
        # Replace negative predictions with zero
        forecast_df[forecast_df < 0] = 0
        # Display forecasted data as CSV
        st.table(forecast_df)

        # Plot line chart for future predictions including the entire dataset
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[selected_column], label='Actual Data', color='blue')
        plt.plot(forecast_df.index, forecast_df[selected_column], label='Future Predictions', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actual Data vs Future Predictions')
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("No predictions available.")
