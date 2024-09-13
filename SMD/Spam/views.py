from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.cache import cache_control
import os
import joblib

model1 = joblib.load(os.path.dirname(__file__) + "\\mySVCModel.pkl")
# Create your views here.

def index(request):
    return render(request, 'index.html')

def CheckSpam(request):
    return render(request, 'output.html')