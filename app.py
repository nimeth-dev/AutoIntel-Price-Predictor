from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# 1. Load the ML model and the Cleaned Data
model = pickle.load(open('CarModelnew.pkl', 'rb'))
data = pd.read_csv('cleaned_car_data.csv')

@app.route('/')
def index():
    # 2. Prepare Dropdown Options
    
    brands = sorted(data['Brand'].unique())
    models = sorted(data['Model'].unique())
    years = sorted(data['YOM'].unique(), reverse=True)
    fuels = sorted(data['Fuel Type'].unique())
    return render_template('index.html', brands=brands, models=models, years=years, fuels=fuels)

@app.route('/predict', methods=['POST'])
def predict():
    # 3. Get Data from HTML Form
    brand = request.form.get('brand')
    car_model = request.form.get('model')
    yom = int(request.form.get('yom'))
    engine_cc = float(request.form.get('engine_cc'))
    fuel = request.form.get('fuel')
    millage = float(request.form.get('millage'))
    
    # 4. Handle "Gear" (Text -> Number)
    # Mapped Manual=1, Automatic=0 in the training phase
    gear_input = request.form.get('gear')
    if gear_input == 'Manual':
        gear = 1
    else:
        gear = 0 

    # 5. Handle "Condition" (Text -> Number)
    #Mapped Used=1, New=0
    condition_input = request.form.get('condition')
    if condition_input == 'Used':
        condition = 1
    else:
        condition = 0 

    # 6. Organize Data for the Model
    
    input_data = pd.DataFrame([[brand, car_model, yom, engine_cc, gear, fuel, millage, condition]], 
                              columns=['Brand', 'Model', 'YOM', 'Engine (cc)', 'Gear', 'Fuel Type', 'Millage(KM)', 'Condition'])

    # 7. Predict
    prediction = model.predict(input_data)
    price = np.round(prediction[0], 2)

    return str(price)

if __name__ == "__main__":
    app.run(debug=True)