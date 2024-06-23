from django.shortcuts import render 
from django.http import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
import os
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .serializers import *
import boto3
from io import StringIO
import warnings
import numpy as np
from smart_open import open
import gzip
from io import BytesIO
from json import loads, dumps


def home(request):
    return HttpResponse("hello world")

warnings.filterwarnings("ignore", category=UserWarning)
class ChartsData(APIView):
    def get(self , request):

        # Create a session with explicit credentials
        print("Hello")

        session = boto3.Session(
            aws_access_key_id= settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key= settings.AWS_SECRET_ACCESS_KEY,
            region_name= settings.AWS_REGION
        ) 
        s3 = session.client('s3')
        print("Client create")
        # Specify the bucket name and the key (file path within the bucket)
        bucket_name = 'fraud-detection-esse'
        key = 'dataset_with_labels.csv'
        print("Start")
        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)
        buffer = []
            # Read the object in chunks
        for i, line in enumerate(response['Body'].iter_lines()):
            if i >= 10:
                break
            buffer.append(line.decode('utf-8'))

        # Join the lines into a single string
        csv_content = '\n'.join(buffer)
        print("CSV content")
        df = pd.read_csv(StringIO(csv_content))
        print("Read file.")

        ################################################################
        # PIE CHART
        loan_status_counts = df['LoanStatus'].value_counts().to_dict()
        business_type_counts = df['BusinessType'].value_counts().to_dict()
        race_counts = df['Race'].value_counts().to_dict()
        ethnicity_counts = df['Ethnicity'].value_counts().to_dict()

        # Combine all the counts into one dictionary
        combined_counts = {**loan_status_counts, **business_type_counts, **race_counts, **ethnicity_counts}

        # Prepare data for Chart.js
        print("Pie Chart complete")
        

        # ################################################################
        # # Prepare data for each bar chart component
        loan_status_counts = df['LoanStatus'].value_counts().to_dict()
        borrower_state_counts = df['BorrowerState'].value_counts().to_dict()
        business_type_counts = df['BusinessType'].value_counts().to_dict()
        race_counts = df['Race'].value_counts().to_dict()
        ethnicity_counts = df['Ethnicity'].value_counts().to_dict()
        gender_counts = df['Gender'].value_counts().to_dict()
        veteran_counts = df['Veteran'].value_counts().to_dict()

        print("bar Chart complete")
        
        # ################################################################

        
        # # LineChart

        # # Convert the 'DateApproved' column to datetime with dayfirst=True
        df['DateApproved'] = pd.to_datetime(df['DateApproved'], dayfirst=True)

        # Sort the dataframe by 'DateApproved'
        df = df.sort_values('DateApproved')

        # Prepare data for DateApproved vs ApprovalDiff
        approval_diff_data = df[['DateApproved', 'ApprovalDiff']].dropna()

        approval_diff_labels = approval_diff_data['DateApproved'].dt.strftime('%Y-%m-%d').tolist()
        approval_diff_values = approval_diff_data['ApprovalDiff'].tolist()

        # Prepare data for DateApproved vs ForgivenessAmount
        forgiveness_amount_data = df[['DateApproved', 'ForgivenessAmount']].dropna()

        forgiveness_amount_labels = forgiveness_amount_data['DateApproved'].dt.strftime('%Y-%m-%d').tolist()
        forgiveness_amount_values = forgiveness_amount_data['ForgivenessAmount'].tolist()

        # Print the JSON data for the line charts
        
        print("line Chart complete")
        

        #################################################################

        # HistogramData

        
        # Function to clean the data by removing infinite values
        def clean_data(data):
            return data[~np.isinf(data)].dropna().tolist()

        # Prepare data for InitialApprovalAmount distribution
        initial_approval_amount_data = clean_data(df['InitialApprovalAmount'])

        # Prepare data for ForgivenessAmount distribution
        forgiveness_amount_data = clean_data(df['ForgivenessAmount'])

        # Prepare data for PROCEED_Per_Job distribution
        proceed_per_job_data = clean_data(df['PROCEED_Per_Job'])

        # Calculate the histogram data using pandas
        initial_approval_hist, initial_approval_bins = pd.cut(initial_approval_amount_data, bins=10, retbins=True, labels=False)
        initial_approval_hist_data = pd.Series(initial_approval_hist).value_counts().sort_index().tolist()

        forgiveness_hist, forgiveness_bins = pd.cut(forgiveness_amount_data, bins=10, retbins=True, labels=False)
        forgiveness_hist_data = pd.Series(forgiveness_hist).value_counts().sort_index().tolist()

        proceed_per_job_hist, proceed_per_job_bins = pd.cut(proceed_per_job_data, bins=10, retbins=True, labels=False)
        proceed_per_job_hist_data = pd.Series(proceed_per_job_hist).value_counts().sort_index().tolist()

        # Create bins labels
        initial_approval_bins_labels = [f"{round(b, 2)} - {round(initial_approval_bins[i+1], 2)}" for i, b in enumerate(initial_approval_bins[:-1])]
        forgiveness_bins_labels = [f"{round(b, 2)} - {round(forgiveness_bins[i+1], 2)}" for i, b in enumerate(forgiveness_bins[:-1])]
        proceed_per_job_bins_labels = [f"{round(b, 2)} - {round(proceed_per_job_bins[i+1], 2)}" for i, b in enumerate(proceed_per_job_bins[:-1])]

        # Print the JSON data for the histograms
        print("Histogram ready")
       

        # ################################################################

        # Heatmap
        numerical_columns = ['InitialApprovalAmount', 'ForgivenessAmount', 'UndisbursedAmount', 'JobsReported']

        # Calculate the correlation matrix
        correlation_matrix = df[numerical_columns].corr()

        # Print the correlation matrix as JSON
        correlation_json = correlation_matrix.to_dict()

        

        print("heat map ready")

        
        # ################################################################
        # Box Plot

        # Prepare the data for box plots
        columns_of_interest = ['InitialApprovalAmount', 'ForgivenessAmount', 'PROCEED_Per_Job']
        categorical_variable = 'LoanStatus'

        # Function to calculate quartiles and outliers
        def calculate_boxplot_data(df, group_col, value_col):
            grouped_data = df.groupby(group_col)[value_col]
            boxplot_data = {}
            for name, group in grouped_data:
                q1 = group.quantile(0.25)
                q3 = group.quantile(0.75)
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                outliers = group[(group < lower_fence) | (group > upper_fence)]
                boxplot_data[name] = {
                    'min': group.min(),
                    'q1': q1,
                    'median': group.median(),
                    'q3': q3,
                    'max': group.max(),
                    'outliers': outliers.tolist()
                }
            return boxplot_data

        # Generate box plot data for each column
        box_plot_data = {}
        for column in columns_of_interest:
            box_plot_data[column] = calculate_boxplot_data(df, categorical_variable, column)
        
        

        print("Box plot ready")

        # ################################################################

        # # Scatter Plot

        initial_approval_vs_forgiveness = df[['InitialApprovalAmount', 'ForgivenessAmount']].dropna().to_dict(orient='list')

        # Prepare data for TOTAL_PROCEED vs PROCEED_Per_Job
        total_proceed_vs_proceed_per_job = df[['TOTAL_PROCEED', 'PROCEED_Per_Job']].dropna().to_dict(orient='list')

        # Print the JSON data for the scatter plots
        

        print("scatter plot ready")
        # ################################################################

        
        # # water fall

        component_columns = [
            'UTILITIES_PROCEED',
            'PAYROLL_PROCEED',
            'MORTGAGE_INTEREST_PROCEED',
            'RENT_PROCEED',
            'REFINANCE_EIDL_PROCEED',
            'HEALTH_CARE_PROCEED',
            'DEBT_INTEREST_PROCEED'
        ]

        # Calculate the sum of each component and the total proceed
        components_sum = df[component_columns].sum().to_dict()
        total_proceed = df['TOTAL_PROCEED'].sum()

        # Prepare the waterfall data
        waterfall_data = {'TOTAL_PROCEED': total_proceed}
        waterfall_data.update(components_sum)

        

        print("Water fall ready")
        
        BarChartRecord = {
            'pie_records': {
                'pie_labels': list(combined_counts.keys()),
                'pie_values': list(combined_counts.values())
            },
            'loan_status': {
                'labels': list(loan_status_counts.keys()),
                'values': list(loan_status_counts.values())
            },
            'borrower_state': {
                'labels': list(borrower_state_counts.keys()),
                'values': list(borrower_state_counts.values())
            },
            'business_type': {
                'labels': list(business_type_counts.keys()),
                'values': list(business_type_counts.values())
            },
            'race': {
                'labels': list(race_counts.keys()),
                'values': list(race_counts.values())
            },
            'ethnicity': {
                'labels': list(ethnicity_counts.keys()),
                'values': list(ethnicity_counts.values())
            },
            'gender': {
                'labels': list(gender_counts.keys()),
                'values': list(gender_counts.values())
            },
            'veteran': {
                'labels': list(veteran_counts.keys()),
                'values': list(veteran_counts.values())
            },
            'line_chart_approval': {
                'labels': approval_diff_labels,
                'values': approval_diff_values
            },
            'line_chart_forgiveness': {
                'labels': forgiveness_amount_labels,
                'values': forgiveness_amount_values
            },
            'histogram_initial': {
                'bins': initial_approval_bins_labels,
                'values': initial_approval_hist_data
            },
            'histogram_forgiveness': {
                'bins': forgiveness_bins_labels,
                'values': forgiveness_hist_data
            },
            'histogram_proceed': {
                'bins': proceed_per_job_bins_labels,
                'values': proceed_per_job_hist_data
            },
            'HeatMapProceed': {
                'bins': proceed_per_job_bins_labels,
                'values': proceed_per_job_hist_data
            }
        }

        print("Data prepared for serialization")

        serializer = BarChartRecordSerializer(data=BarChartRecord)

        if serializer.is_valid():
            return Response({'status': 200, 'message': 'Charts Fetch Sucessfully', 'payload': serializer.data}, status = status.HTTP_200_OK )       
        print("Serializer errors:")
        print(serializer.errors) 
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FileReaderAPIView(APIView):
    def get(self, request):
        try:
            # Create a session with explicit credentials
            session = boto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            s3 = session.client('s3')
            print("S3 client created successfully")

            # Specify the bucket name and the key (file path within the bucket)
            bucket_name = 'fraud-detection-esse'
            key = 'dataset_with_labels.csv'

            # Get the object from S3
            response = s3.get_object(Bucket=bucket_name, Key=key)
            print("S3 object retrieved successfully")

            # Create a buffer to store the lines
            buffer = []

            # Read the object in chunks
            for i, line in enumerate(response['Body'].iter_lines()):
                if i >= 10:
                    break
                buffer.append(line.decode('utf-8'))

            # Join the lines into a single string
            csv_content = '\n'.join(buffer)
            print("CSV content read successfully")

            if csv_content:
                df = pd.read_csv(StringIO(csv_content))
                json_data = df.to_json(orient='records')
                parsed = loads(json_data)
                serializer = JSONResponseSerializer(data={'json_data': parsed})
                serializer.is_valid(raise_exception=True)
                print("Data serialized successfully")
                return Response({'status': 200, 'message': 'File read successfully', 'payload': serializer.data}, status=status.HTTP_200_OK)

        except Exception as e:
            print("An error occurred:", str(e))
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, content_type='application/json')


class gpt(APIView):
    def get(self, request):
        x = "1234"
        serializer = barser(data={'p': x})
        if serializer.is_valid():
            return Response({'status': 201, 'message': 'User created successfully', 'payload': serializer.data}, status = status.HTTP_201_CREATED )       

        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import boto3
import joblib
from io import BytesIO
# Load the model from S3

session = boto3.Session(
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)
s3 = session.client('s3')





    
# Define function to encode categorical features
def encode(data):
    encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    return data

def XGBoost_model(csv_row):
    input_data = pd.DataFrame([csv_row])
    input_data_encoded = encode(input_data)
    bucket_name = 'fraud-detection-esse'
    key = 'XGBoost_model.joblib'
    response = s3.get_object(Bucket=bucket_name, Key=key)
    model_bytes = response['Body'].read()
    model = joblib.load(BytesIO(model_bytes))
    prediction = model.predict(input_data_encoded)
    return prediction

def RandomForest_model(csv_row):
    input_data = pd.DataFrame([csv_row])
    input_data_encoded = encode(input_data)
    bucket_name = 'fraud-detection-esse'
    key = 'RandomForest_model.joblib'
    response = s3.get_object(Bucket=bucket_name, Key=key)
    model_bytes = response['Body'].read()
    model = joblib.load(BytesIO(model_bytes))
    prediction = model.predict(input_data_encoded)
    return prediction

def LogisticRegression_model(csv_row):
    input_data = pd.DataFrame([csv_row])
    input_data_encoded = encode(input_data)
    bucket_name = 'fraud-detection-esse'
    key = 'LogisticRegression_model.joblib'
    response = s3.get_object(Bucket=bucket_name, Key=key)
    model_bytes = response['Body'].read()
    model = joblib.load(BytesIO(model_bytes))
    prediction = model.predict(input_data_encoded)
    return prediction

def LightGBM(csv_row):
    input_data = pd.DataFrame([csv_row])
    input_data_encoded = encode(input_data)
    bucket_name = 'fraud-detection-esse'
    key = 'LightGBM.joblib'
    response = s3.get_object(Bucket=bucket_name, Key=key)
    model_bytes = response['Body'].read()
    model = joblib.load(BytesIO(model_bytes))
    prediction = model.predict(input_data_encoded)
    return prediction

class FraudPredictionAPIView(APIView):
    def post(self, request):
        serializer = FraudPredictionSerializer(data=request.data)
        if serializer.is_valid():
            prediction1 = XGBoost_model(serializer.validated_data)
            prediction2 = LightGBM(serializer.validated_data)
            prediction3 = RandomForest_model(serializer.validated_data)
            prediction4 = LogisticRegression_model(serializer.validated_data)
            if prediction1 == 0:
                res1 = "Fraud"
            else:
                res1 = "No Fraud"
                
            if prediction2 == 0:
                res2 = "Fraud"
            else:
                res2 = "No Fraud"

            if prediction3 == 0:
                res3 = "Fraud"
            else:
                res3 = "No Fraud"

            if prediction4 == 0:
                res4 = "Fraud"
            else:
                res4 = "No Fraud"

            result = {
                'XGBoost_model' : res1,
                'LightGBM' : res2,
                'RandomForest_model' : res3,
                'LogisticRegression_model' : res4,

            }
            return Response({'status': 200, 'message': 'Prediction successfully', 'payload': result}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

