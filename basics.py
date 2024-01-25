import streamlit as st 
import sklearn as sk 
from azureml.core import Workspace
from azureml.core.model import Model 

st.write("""
#         Radim         """)

import json
import joblib
from pathlib import Path
from azureml.core.workspace import Workspace, Webservice
 

import urllib.request
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
data = {}

body = str.encode(json.dumps(data))

url = 'http://413e8fdd-69e6-4c3c-a3e2-272da7e5eb89.westeurope.azurecontainer.io/score'


headers = {'Content-Type':'application/json'}

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


service_name = 'firstdeploy'
ws = Workspace.get(
    name='mobile_price_classification_ws',
    subscription_id='6b585342-4af0-46e8-8079-f174865ff233',
    resource_group='mobile_price_classification_rg'
)

model_name = "firstmodel"
model = Model(ws, model_name)
model_path = Model.get_model_path(model_name)
model = joblib.load(model_path)

service = Webservice(ws, service_name)
sample_file_path = "C:/Users/pauli/Downloads/firstmodel_/_samples.json"
 
with open(sample_file_path, 'r') as f:
    sample_data = json.load(f)
score_result = service.run(json.dumps(sample_data))
st.write((f'Inference result = {score_result}'))

import json
from azureml.core import Workspace, Webservice

# Dohvaćanje Azure ML web servisa
service = Webservice(ws, service_name)

# Postavljanje Streamlit aplikacije
st.title("Azure ML Streamlit App")

# Korisnički unos značajki
ram_size = st.number_input("Unesite veličinu rama:", min_value=0.0, max_value=32.0, step=1.0, value=8.0)
clock_speed = st.number_input("Unesite brzinu procesora (GHz):", min_value=0.0, max_value=4.0, step=1.0, value=2.0)
battery_power=st.number_input("Unesite veličinu bat power", min_value=0.0, max_value=32.0, step=1.0, value=0.0)
bluetooth=st.number_input("Unesite veličinu bluetooth:", min_value=0.0, max_value=32.0, step=1.0, value=0.0)
dual_sim=st.number_input("Unesite veličinu dualsim:", min_value=0.0, max_value=32.0, step=1.0, value=0.0)
front_camera=st.number_input("Unesite veličinu frontcam", min_value=0.0, max_value=32.0, step=1.0, value=0.0)
four_g=st.number_input("Unesite veličinu 4G:", min_value=0.0, max_value=32.0, step=1.0, value=0.0)
internal_memory=st.number_input("Unesite veličinu rama internal:", min_value=0.0, max_value=32.0, step=1.0, value=1.0)
mobile_weight=st.number_input("Unesite veličinu rama weight:", min_value=0.0, max_value=32.0, step=1.0, value=1.0)
num_cores=st.number_input("Unesite veličinu rama numcores:", min_value=0.0, max_value=32.0, step=1.0, value=4.0)
primary_camera=st.number_input("Unesite veličinu rama primcam:", min_value=0.0, max_value=32.0, step=1.0, value=1.0)
screen_height=st.number_input("Unesite veličinu rama screen height:", min_value=0.0, max_value=32.0, step=1.0, value=1.0)
screen_width=st.number_input("Unesite veličinu rama screen width:", min_value=0.0, max_value=32.0, step=1.0, value=1.0)
three_g=st.number_input("Unesite veličinu rama 3G:", min_value=0.0, max_value=32.0, step=1.0, value=1.0)
touch_screen=st.number_input("Unesite veličinu rama touch screen:", min_value=0.0, max_value=32.0, step=1.0, value=0.0)
wifi=st.number_input("Unesite veličinu rama wifi:", min_value=0.0, max_value=32.0, step=1.0, value=1.0)
# Dodajte druge značajke prema potrebi

# Gumb za pokretanje predviđanja
if st.button("Pokreni predviđanje"):
    # Kreiranje objekta s unesenim značajkama
    user_input = {
        "ram": ram_size,
        "clock_speed": clock_speed,
        "battery_power":battery_power,
        "bluetooth":bluetooth,
        "dual_sim":dual_sim,
        "front_camera":front_camera,
        "four_g":four_g,
        "internal_memory":internal_memory,
        "mobile_weight":mobile_weight,
        "num_cores":num_cores,
        "primary_camera":primary_camera,
        "screen_height":screen_height,
        "screen_width":screen_width,
        "three_g":three_g,
        "touch_screen":touch_screen,
        "wifi":wifi,
        # Dodajte druge značajke prema potrebi
    }

    # Pokretanje predviđanja pomoću Azure ML web servisa
    score_result = service.run(json.dumps(user_input))

    # Prikazivanje rezultata predviđanja
    st.subheader("Rezultat predviđanja:")
    st.json(score_result)

    # Odabir klase s najvećom vjerojatnošću (za višeklasnu klasifikaciju)
    predicted_class = max(score_result, key=score_result.get)
    st.subheader(f"Predviđena klasa: {predicted_class}")

