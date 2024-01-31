import streamlit as st 
import sklearn as sk 
from azureml.core import Workspace
from azureml.core.model import Model 
import json
import pandas as pd
import numpy as np 
from azureml.core import Workspace, Webservice
import matplotlib.pyplot as plt 
st.write("""
#         Radim         """)

import json
import joblib
from pathlib import Path
from azureml.core.workspace import Workspace, Webservice
 

import urllib.request
import os
import ssl

import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {}

body = str.encode(json.dumps(data))

url = 'http://bf56dad9-dee0-46cf-8f7a-fcfeaf154bc7.westeurope.azurecontainer.io/score'
# Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = 'Z9s0s9UM8FIIcoOMT3ZyOyNt7zeq4utB'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")


headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))


service_name = 'mobileclassificationdeploy'
ws = Workspace.get(
    name='mobile_price_classification_ws',
    subscription_id='6b585342-4af0-46e8-8079-f174865ff233',
    resource_group='mobile_price_classification_rg'
)

model_name = "amlstudio-mobileclassification"
model = Model(ws, model_name)

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

#test_dataset = Dataset.get_by_name(ws, name='test_dataset')
#test_dataset.to_pandas_dataframe()
import json

# Function to load JSON data into a Pandas DataFrame
def json_to_dataframe(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        test_dataset= pd.DataFrame(data)
        return test_dataset

# Example usage
test_dataset=json_to_dataframe('scheme.json')
#df = json_to_dataframe('your_json_file.json')
#bat_pow=test_dataset['battery_power']
# Dohvaćanje Azure ML web servisa

service = Webservice(ws, service_name)
#score_testdata_result = service.run(json.dumps(test_dataset))


# Postavljanje Streamlit aplikacije
st.title("Azure ML Streamlit App")

# Korisnički unos značajki

battery_power= st.number_input("battery_power: ", min_value=0, max_value=3200, step=100, value=100)
bluetooth= st.selectbox("bluetooth: ", (0, 1))
clock_speed= st.number_input("clock_speed: ",min_value=0.5, max_value=3.0, step=0.1, value=0.5)
dual_sim= st.selectbox("dual_sim: ", (0, 1))
front_camera= st.number_input("front_camera: ", min_value=0, max_value=19, step=1, value=0)
four_g= st.selectbox("four_g: ", (0, 1))
internal_memory= st.number_input("internal memory: ", min_value=2, max_value=64, step=1, value=2)
mobile_depth=st.number_input("mobile_depth: ", min_value=0.1, max_value=1.0, step=0.1, value=0.1)
mobile_weight=st.number_input("mobile_weight: ", min_value=80, max_value=200, step=1, value=80)
num_cores=st.number_input("num_cores: ", min_value=1, max_value=8, step=1, value=1)
primary_camera=st.number_input("primary_camera: ", min_value=0, max_value=20, step=1, value=0)
px_resolution_height= st.number_input("px_resolution_height: ", min_value=500, max_value=1960, step=1, value=500 )
px_resolution_width=st.number_input("px_resolution_width: ", min_value=500, max_value=1998, step=1, value=500 )
ram=st.number_input("ram: ", min_value=256, max_value=3998, step=1, value=256 )
screen_height=st.number_input("screen_height: ", min_value=5, max_value=19, step=1, value=5 )
screen_width=st.number_input("screen_width: ", min_value=1, max_value=18, step=1, value=1 )
talk_time=st.number_input("talk_time: ", min_value=2, max_value=20, step=1, value=2 )
three_g=st.selectbox("three_g: ",(0, 1))
touch_screen=st.selectbox("touch_screen: ", (0, 1))
wifi=st.selectbox("wifi: ", (0, 1))

# Dodajte druge značajke prema potrebi

# Gumb za pokretanje predviđanja
if st.button("Pokreni predviđanje"):
    data={    # Kreiranje objekta s unesenim značajkama
        "Inputs":{
            "WebServiceInput0":[{
                                        "battery_power": battery_power,
                                        "bluetooth": bluetooth,
                                        "clock_speed": clock_speed,
                                        "dual_sim": dual_sim,
                                        "front_camera": front_camera,
                                        "four_g": four_g,
                                        "internal_memory": internal_memory,
                                        "mobile_depth": mobile_depth,
                                        "mobile_weight": mobile_weight,
                                        "num_cores": num_cores,
                                        "primary_camera": primary_camera,
                                        "px_resolution_height": px_resolution_height,
                                        "px_resolution_width": px_resolution_width,
                                        "ram": ram,
                                        "screen_height": screen_height,
                                        "screen_width": screen_width,
                                        "talk_time": talk_time,
                                        "three_g": three_g,
                                        "touch_screen": touch_screen,
                                        "wifi": wifi,
            }
        ]},
        "GlobalParameters": {}
    }

    # Pokretanje predviđanja pomoću Azure ML web servisa
    score_result = service.run(json.dumps(data))
    #score_testdata_result = service.run(json.dumps(test_dataset))
    score_testdata_result = service.run(json.dumps(data))
    st.write(score_testdata_result)
    
    # Prikazivanje rezultata predviđanja
    st.subheader("Rezultat predviđanja:")
    st.json(score_result)
    # Dohvaćanje relevantnih dijelova
    results = score_result.get("Results", {})
    web_service_output = results.get("WebServiceOutput0", [])
    first_result = web_service_output[0] if web_service_output else {}

    # Dohvaćanje pojedinačnih značajki
    price_range = int(first_result.get("Price range"))
    prob_0 = first_result.get("Scored Probabilities 0")
    prob_1 = first_result.get("Scored Probabilities 1")
    prob_2 = first_result.get("Scored Probabilities 2")
    prob_3 = first_result.get("Scored Probabilities 3")

    # Ispisivanje vrijednosti
    st.write("Battery Power:", battery_power) 
    st.write("Price range:", price_range)
    st.write("Scored Probabilities 0:", prob_0)
    st.write("Scored Probabilities 1:", prob_1)
    st.write("Scored Probabilities 2:", prob_2)
    st.write("Scored Probabilities 3:", prob_3)

    datad = {'x': ['Scored Probabilities 0:', 'Scored Probabilities 1:', 'Scored Probabilities 2:', 'Scored Probabilities 3:'], 'y': [prob_0, prob_1, prob_2, prob_3]}
    df = pd.DataFrame(datad)

    # Prikazivanje trakastog grafikona
    st.bar_chart(df.set_index('x'))


if st.button("Pokreni test predviđanje"):
    with open("C:/Users/pauli/streamlit/app/csvjson.json", 'r') as file:
        testdatadata = json.load(file)

    score_test_result = service.run(json.dumps(testdatadata))
    st.write(score_test_result)
    
    allinput = score_test_result.get("Input", {})
    inputall = allinput.get("WebServiceInput0", [])
    allinputresault= inputall[::] if inputall else {}
    battery_power = [item['battery_power'] for item in allinputresault]
    st.write(battery_power)

    resultstest = score_test_result.get("Results", {})
    web_service_output_test = resultstest.get("WebServiceOutput0", [])
    all_result_test = web_service_output_test[::] if web_service_output_test else {}
    st.write(all_result_test)
    price_range = [item['Price range'] for item in all_result_test]
    st.write(price_range)

    # Izvlačenje podataka iz JSON datoteke
    if len(testdatadata) == len(price_range):
        # Kreiranje DataFrame-a
        dff = pd.DataFrame({
                'battery_power': battery_power,
                'price_range': price_range 

        })
        # Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(dff['battery_power'], dff['price_range'], alpha=0.5)
        plt.title('Scatter Plot of Battery Power vs Price Range')
        plt.xlabel('Battery Power')
        plt.ylabel('Price Range')
        plt.grid(True)
        plt.show()
        st.pyplot(plt)


   