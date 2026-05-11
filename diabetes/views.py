from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'diabetes/overview.html')

def predict(request):
    return render(request, 'diabetes/predict.html')

def evaluation(request):
    return render(request, 'diabetes/evaluation.html')

def model_training(request):
    return render(request, 'diabetes/model_training.html')