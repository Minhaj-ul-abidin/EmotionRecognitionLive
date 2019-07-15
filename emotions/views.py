from django.shortcuts import render
from django.shortcuts import render

# Create your views here.
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
import numpy as np

from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser

from .Utilities.emotion_recog_api import get_objects
import json
import cv2
# Create your views here.
@api_view(['POST'])
@parser_classes((FormParser, MultiPartParser))
def find_emotion(request, format=None):
    try:
        print("Name",request.data.get('image').name)
        image_data = request.data['image']
        np_img = np.fromstring(image_data.read(),np.uint8)
        print(cv2.imdecode(np_img,3))
        # print(get_objects(np_img))
        return HttpResponse(get_objects(cv2.imdecode(np_img,3)))
    except ValueError as e:
        print(e)
        return Response(e,status.HTTP_400_BAD_REQUEST)