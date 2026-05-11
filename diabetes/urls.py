from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.index, name='overview'),
    path('overview/', views.index, name='overview'),
    path('predict/', views.predict, name='predict'),
    path('model_training/', views.model_training, name='model_training'),
    path('evaluation/', views.evaluation, name='evaluation'),
]
