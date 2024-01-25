import streamlit as st 
import sklearn as sk 
from azureml.core import Workspace
from azureml.core.model import Model 

st.write("""
#         Radim         """)

import json
from pathlib import Path
from azureml.core.workspace import Workspace, Webservice
 
service_name = 'firstdeploy'
ws = Workspace.get(
    name='mobile_price_classification_ws',
    subscription_id='6b585342-4af0-46e8-8079-f174865ff233',
    resource_group='mobile_price_classification_rg'
)
service = Webservice(ws, service_name)
sample_file_path = '_samples.json'
 
with open(sample_file_path, 'r') as f:
    sample_data = json.load(f)
score_result = service.run(json.dumps(sample_data))
print(f'Inference result = {score_result}')
