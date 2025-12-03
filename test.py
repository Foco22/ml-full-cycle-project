##################################################
# Archivo pipeline.py
# Realiza el registro del archivo "inference.py" como un pipeline de Azure ML, desactivando la versión anterior del pipeline
# Además, registra y actualiza el id del pipeline de AML en el secret de keyvault llamado "aml-pipelineid-pred-<NOMBRE_DEL_CASO_DE_USO>"
##################################################
import os
from datetime import datetime
import json
import urllib.request
import ssl
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import locale
import sqlalchemy
import json
from lxml import etree

def fetch_data(url, headers, context):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, context=context) as response:
        return response.read()
    

rg_env = 'dev'
Fecha_Inicio = datetime.datetime(1900, 1, 1)
anno_descarga = Fecha_Inicio.strftime("%Y")
mes_descarga = Fecha_Inicio.strftime("%m")
with open('parameters.json', 'r') as f:
    parameters = json.load(f)

driver = parameters['sql']['driver']
server = parameters['sql']['server_{}'.format(rg_env)]
database = parameters['sql']['database']
username = parameters['sql']['username']
password = parameters['sql']['password_{}'.format(rg_env)]
api_key = parameters['api_key']
schema = parameters['sql']['schema_{}'.format(rg_env)]

ruta_usd = f'https://api.cmfchile.cl/api-sbifv3/recursos_api/dolar/posteriores/{anno_descarga}/{mes_descarga}?apikey={api_key}&formato=xml'

# Create an unverified SSL context
context = ssl._create_unverified_context()

# Add User-Agent header
headers = {'User-Agent': 'Mozilla/5.0'}

download_USD = fetch_data(ruta_usd, headers, context)

soup_usd = BeautifulSoup(download_USD, features="xml")

# Extracting the data
fechas_usd = soup_usd.find_all('Fecha')
valores_usd = soup_usd.find_all('Valor')

data_usd = []

# Loop to store the data in a list named 'data'
for i in range(0, len(fechas_usd)):
    rows = [fechas_usd[i].get_text(), valores_usd[i].get_text()]
    data_usd.append(rows)

df_usd = pd.DataFrame(data_usd, columns=['Fecha', 'usdclp_obs'])

print(df_usd.head())
