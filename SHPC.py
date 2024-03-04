#Este código es el intento de programar una versión no paralelizada del
#algoritmo SHPC desarrollado por sofía, carlos y ana
#El autor de este código es Johan, y espero que sirva en un futuro dado


#Carga de librerias
import cv2
import numpy as np
from numpy import asarray
from Funciones import *
import spicy as sp
import math as mt

#Funcion de reconstrucción

def tiro(holo,fx_0,fy_0,fx_tmp, fy_tmp,lamb,M,N,dx,dy,k,m,n):
    
    #Calculo de los angulos de inclinación

    theta_x=mt.asin((fx_0 - fx_tmp) * lamb /(M*dx))
    theta_y=mt.asin((fy_0 - fy_tmp) * lamb /(N*dy))

    #Creación de la fase asociada

    fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
    fase1=fase
    holo=holo*fase
    
    fase = np.angle(holo, deg=False)
    min_val = np.min(fase)
    max_val = np.max(fase)
    fase = (fase - min_val) / (max_val - min_val)
    threshold_value = 0.2
    fase = np.where(fase > threshold_value, 1, 0)
    value=np.sum(fase)
    return value, fase1

archivo = "Imagenes/frames/001.bmp"
replica = lectura(archivo)
U = asarray(replica)
N, M = U.shape
#Parametros del montaje

#Trabajemos todo en metros o indica porfa que unidades usas
#Que si no me enredo

# dx y dy como dice el codigo jajaja
dx = 3.75
dy = 3.75
lamb = 0.633
k= 2*np.pi/lamb
Fox= M/2
Foy= N/2
cuadrante=1
# pixeles en el eje x y y de la imagen de origen
x = np.arange(0, M, 1)
y = np.arange(0, N, 1)

#Un meshgrid para la paralelizacion
m, n = np.meshgrid(x - (M/2), y - (N/2))

G=3

#Definiendo cuadrantes, solo calidad
primer_cuadrante= np.zeros((N,M))
primer_cuadrante[0:round(N/2 - (N*0.1)),round(M/2 + (M*0.1)):M]=1
segundo_cuadrante= np.zeros((N,M))
segundo_cuadrante[0:round(N/2 -(N*0.1)),0:round(M/2 - (M*0.1))]=1
tercer_cuadrante= np.zeros((N,M))
tercer_cuadrante[round(N/2 +(N*0.1)):N,0:round(M/2 - (M*0.1))]=1
cuarto_cuadrante= np.zeros((N,M))
cuarto_cuadrante[round(N/2 +(N*0.1)):N,round(M/2 + (M*0.1)):M]=1

#Ahora a tirar fourier

fourier=np.fft.fftshift(sp.fft.fft2(np.fft.fftshift(U)))
b=amplitud(fourier)
if(cuadrante==1):
    fourier=primer_cuadrante*fourier
if(cuadrante==2):
    fourier=segundo_cuadrante*fourier
if(cuadrante==3):
    fourier=tercer_cuadrante*fourier
if(cuadrante==4):
    fourier=cuarto_cuadrante*fourier
a=amplitud(fourier)
#Calculamos la amplitud del espectro de fourier



#Transformada insversa de fourier
fourier=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier)))
mapa=fourier
#Encontramos la posición en x y y del máximo en el espacio de Fourier
pos_max = np.unravel_index(np.argmax(a, axis=None), a.shape)


#Ahora viene definición de parametros 

paso=0.2
fin=0
fx=pos_max[1]
fy=pos_max[0]
print(fx)
print(fy)
G_temp=G
suma_maxima=0


while fin==0:
    temp=0
    i=0
    j=0
    frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
    frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
    for i in range(len(frec_esp_y)):
        for j in range(len(frec_esp_x)):
            fx_temp=frec_esp_x[j]
            fy_temp=frec_esp_y[i]
            temp, faserina=tiro(fourier,Fox,Foy,fx_temp,fy_temp,lamb,M,N,dx,dy,k,m,n)
            if(temp>suma_maxima):
                x_max_out = fx_temp
                y_max_out = fy_temp
                suma_maxima = temp
    G_temp = G_temp - 1
    
    if(x_max_out == fx):
        if(y_max_out ==fy):
            fin=1
    fx=x_max_out
    fy=y_max_out

theta_x=mt.asin((Fox - fx) * lamb /(M*dx))
theta_y=mt.asin((Foy - fy) * lamb /(N*dy))
print(fx)
print(fy)
fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
holo=fourier*fase
fase1=fase
fase = np.angle(holo, deg=False)
min_val = np.min(fase)
max_val = np.max(fase)
fase = 255*(fase - min_val) / (max_val - min_val)
titulo = "a"
ejex_1 = "b"
ejey_1 = "c"
mostrar((fase),titulo,ejex_1,ejey_1)
