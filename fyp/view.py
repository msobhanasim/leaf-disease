#this file contains views
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import json

from rest_framework.parsers import FileUploadParser
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.response import Response
from rest_framework import status , generics , mixins,filters,viewsets
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
import boto3

import matplotlib.image as mpimg
import numpy as np
import boto3

class Predict(APIView):
    
    def post(self,request, *args, **kwargs):
        
        imageName=request.data.get('image_name')
        filename = "{}".format(imageName)

        folder_name = 'Java_Upload/'
        
        s3 = boto3.resource('s3', region_name=settings.AWS_REGION , aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
             aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)

        bucket = s3.Bucket(settings.AWS_STORAGE_BUCKET_NAME)

        object_1 = bucket.Object(folder_name + filename)
        object_1.download_file(filename)

        img=mpimg.imread(filename)
        
        client = boto3.client('runtime.sagemaker', 
        region_name=settings.AWS_REGION, 
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
        
        with open(filename, 'rb') as f:
            payload = f.read()
            payload = bytearray(payload)
        
        response = client.invoke_endpoint(EndpointName=settings.AWS_SAGEMAKER_ENDPOINT_NAME, 
                                        ContentType='application/x-image', 
                                        Body=payload)
        result = response['Body'].read()
       
        # result will be in json format and convert it to ndarray
        result = json.loads(result)
       
        index = np.argmax(result)
        object_categories = ['GLS', 'CR', 'NLB', 'Healthy' , 'Extra_Class']
       
       
        acc = '{}'.format(round(result[index],2))
        label = '{}'.format(object_categories[index])

        fnl_JSON = {
                    'accuracy': acc,
                    'label': label
                }

        # convert into JSON:
        output = json.dumps(fnl_JSON)
        print(output)
     
        return Response(fnl_JSON)
                    


