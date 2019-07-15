from django.urls import path

from . import views

urlpatterns = [
    path('findemotions/', views.find_emotion, name='find_emotion'),

]