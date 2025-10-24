from django.urls import path 
from . import views

urlpatterns=[
    path('',views.home,name="home"),
    path('modulo1/',views.modulo1,name='modulo1'),
    path('modulo2/',views.modulo2,name='modulo2'),
    path('modulo3/',views.modulo3,name='modulo3'),
    path('simplex/',views.modulo1_simplex,name='simplex'),
    path('grafico/',views.modulo_grafico,name='grafico'),
    path('gran_m/',views.gran_m,name="gran_m"),
    path('dos_fases/',views.dos_fases,name="dos_fases"),
    path('modulo_dual/', views.modulo_dual, name='modulo_dual'),
    path('modulo_inventario/', views.modulo_inventario, name='modulo_inventario'),
]