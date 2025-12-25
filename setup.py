"""
Setup script for exchange rate prediction ML project.
"""

from setuptools import setup, find_packages

setup(
    name='exchange-rate-ml',
    version='1.0.0',
    description='Exchange Rate Prediction ML Model (model_balanced)',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'xgboost==2.0.3',
        'scikit-learn==1.4.0',
        'pandas==2.2.0',
        'numpy==1.26.3',
        'pyyaml==6.0.1',
        'google-cloud-bigquery==3.14.1',
        'google-cloud-storage==2.14.0',
    ],
)