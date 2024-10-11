from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start_trading/', views.start_trading, name='start_trading'),
    path('stop_trading/', views.stop_trading, name='stop_trading'),
    path('bot_status/', views.bot_status, name='bot_status'),
]