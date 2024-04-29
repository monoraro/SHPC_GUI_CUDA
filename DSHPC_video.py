"""
Este código permite ingresar una carpeta de hologramas y reconstruirlos

finalmente guarda el resultado en video
"""

#Carga de librerias
import cv2
import numpy as np
from numpy import asarray
import math as mt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from pycuda.reduction import ReductionKernel
import os
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from Funciones import *
import time
import imageio

#Hay más cosas que debo considerar, en general siento que
#Esto lo saqué del de kreuzer porque al final es la misma vaina

#Cargamos el archivo del holograma
# Ruta a los frames
carpeta = './Imagenes/40X'
carpeta2 = 'Imagenes/40X/'
# Obtener la lista de archivos en la carpeta
archivos_en_carpeta = os.listdir(carpeta)

# Ordenar la lista de archivos alfabéticamente
archivos_ordenados = sorted(archivos_en_carpeta)
replica = archivos_ordenados[0]
archivo = carpeta2+str(replica)
replica = lectura(archivo)
replica=ajuste_tamano(replica)

tiempo_inicial = time.time()
U = asarray(replica)

U = np.asarray(replica)  # Esto asume que el frame es de 3 canales y selecciona solo el canal 0

N, M = U.shape
# dx y dy como dice el codigo jajaja
dx = 3.75
dy = 3.75
lamb = 0.633
k = 2*mt.pi/lamb
Fox = M/2
Foy = N/2
cuadrante = 1
G = 4
threso = 0.2
# pixeles en el eje x y y de la imagen de origen
x = np.arange(0, M, 1)
y = np.arange(0, N, 1)

#Un meshgrid para la paralelizacion
m, n = np.meshgrid(x - (M/2), y - (N/2))

#Esta variable sirve para inicializar el valor mínimo de la suma
suma_max = np.array([[0]]) 
#Definicion de tipos de variables compatibles con cuda
U = U.astype(np.float32)
m = m.astype(np.float32)
n = n.astype(np.float32)
suma_max = suma_max.astype(np.float32)

#Variables definidas a la gpu
U_gpu = gpuarray.to_gpu(U)
m_gpu = gpuarray.to_gpu(m)
n_gpu = gpuarray.to_gpu(n)
suma_max = gpuarray.to_gpu(suma_max)

if(cuadrante==1):
    primer_cuadrante= np.zeros((N,M))
    primer_cuadrante[0:round(N/2 - (N*0.1)),round(M/2 + (M*0.1)):M] = 1
    primer_cuadrante = primer_cuadrante.astype(np.float32)
    cuadrante_gpu = gpuarray.to_gpu(primer_cuadrante)
if(cuadrante==2):
    segundo_cuadrante= np.zeros((N,M))
    segundo_cuadrante[0:round(N/2 -(N*0.1)),0:round(M/2 - (M*0.1))] = 1
    segundo_cuadrante = segundo_cuadrante.astype(np.float32)
    cuadrante_gpu = gpuarray.to_gpu(segundo_cuadrante)

if(cuadrante==3):
    tercer_cuadrante= np.zeros((N,M))
    tercer_cuadrante[round(N/2 +(N*0.1)):N,0:round(M/2 - (M*0.1))] = 1
    tercer_cuadrante = tercer_cuadrante.astype(np.float32)
    cuadrante_gpu = gpuarray.to_gpu(tercer_cuadrante)

if(cuadrante==4):
    cuarto_cuadrante= np.zeros((N,M))
    cuarto_cuadrante[round(N/2 +(N*0.1)):N,round(M/2 + (M*0.1)):M] = 1
    cuarto_cuadrante = cuarto_cuadrante.astype(np.float32)
    cuadrante_gpu = gpuarray.to_gpu(cuarto_cuadrante)

mod = SourceModule("""
#include <math.h>
#include <stdio.h>
#include <cuComplex.h>
#include <limits>

                   /*FUNCIONES PARALELIZADAS DEL ALGORITMO*/

__global__ void fase_refe(cuComplex *holo, cuComplex *holo2, cuComplex *ref, float *m, float *n,  int N, int M, float k, float fox, float foy, float fx, float fy, float lamb, float dx, float dy)
{
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = (col * N) + fila;

    // Precalcula senos y cosenos
    double temp_x = (fox - fx) * lamb / (M * dx);
    double temp_y = (foy - fy) * lamb / (N * dx);
    double tx = asin(temp_x);
    double ty = asin(temp_y);
    float sin_tx = sin(tx);
    float sin_ty = sin(ty);

    // Cálculo de la fase
    float temporal = k * ((sin_tx * m[i2] * dx) + (sin_ty * n[i2] * dy));
    float cos_temporal = cos(temporal);
    float sin_temporal = sin(temporal);

    // Guardar en memoria global
    ref[i2].x = cos_temporal;
    ref[i2].y = sin_temporal;

    // Multiplicación directa en memoria global
    holo2[i2] = cuCmulf(holo[i2], ref[i2]);

}
              

                   /*Función paralelizada para encontrar el máximo junto con su respectiva posiciAmplitudón
                
                   //Como creo vectores temporales jajaja
                    //Ya lo programaron en pycuda jajaja, Pero la teoria es interesante, es como un arbol genealógico al revéz
                    */
__global__ void coordenadas_maximo(float *matrix, int rows, int cols, int *max_position){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid == 0)
    {
        float max_value = -1e9;  // Valor inicial muy pequeño
        int max_index = -1;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i * cols + j] > max_value)
                {
                    max_value = matrix[i * cols + j];
                    max_index = i * cols + j;
                }
            }
        }

        *max_position = max_index;
    }
}
__global__ void reseteo(cuComplex *holo,int N, int M){
    //Estos son parametros necesarios para definición
    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;
    int i2= ((fila*M)+col);
    holo[i2].x=0;
    holo[i2].y=0;           
}

                   /*FUNCION DE SHIFTEO CUANDO SE TIENE LA INFORMACIÓN EN 2 COMPLEX 64 */
__global__ void fft_shift(cuComplex *final,cuComplex *dest_gpu, int N, int M)
{
	int n2 = N / 2;
	int m2 = M / 2;

	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
	int fila2 = fila + n2;
	int col2 = col + m2;
    
    final[(fila2*M + col2)].x = dest_gpu[(fila*M+col)].x;  //Guardo el primer cuadrante
	final[(fila*M+col)].x = dest_gpu[(fila2*M+col2)].x;  //en el primer cuadrante estoy poniendo lo que hay en el tercero
    final[(fila2*M + col2)].y = dest_gpu[(fila*M+col)].y;  //Lo mismo pero para los imaginarios
	final[(fila*M+col)].y = dest_gpu[(fila2*M+col2)].y;
    
    final[(fila*M + col2)].x = dest_gpu[(fila2*M+col)].x;  //Guardo el segundo cuadrante
	final[(fila2*M+col)].x = dest_gpu[(fila*M+col2)].x;  //en el segundo cuadrante estoy poniendo lo que hay en el tercer cuadrante
    final[(fila*M + col2)].y = dest_gpu[(fila2*M+col)].y;  //Lo mismo pero para los imaginarios
	final[(fila2*M+col)].y = dest_gpu[(fila*M+col2)].y;  
}

                   /*FUNCION DE SHIFTEO SE TIENE LA INFORMACIÓN NO COMPLEJA */              
__global__ void fft_shift_var_no_compleja(cuComplex *final,float *U_gpu, int N, int M)
{
	int n2 = N / 2;
	int m2 = M / 2;

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x+ threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int fila2 = blockIdx.x*blockDim.x + threadIdx.x + n2;
	int col2 = blockIdx.y*blockDim.y + threadIdx.y + m2;
    final[fila*M+col].x = U_gpu[((fila2*M) + (col2))];
    final[fila*M+col].y = 0;
    final[fila2*M+col2].x = U_gpu[((fila*M) + (col))];
    final[fila2*M+col2].y = 0;
    final[fila*M+col2].x = U_gpu[((fila2*M) + (col))];
    final[fila*M+col2].y = 0;
    final[fila2*M+col].x =U_gpu[((fila*M) + (col2))];
    final[fila2*M+col].y = 0;
}

__global__ void thresholding(float *image, int N, int M, float threshold)
{
    int fila = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
                   
    int i2= ((fila*M)+col);
    image[i2] = (image[i2] > threshold) ? 1.0 : 0.0;
}
                   /*mascaras para una imagen dada */                  
__global__ void mascara_1er_cuadrante(cuComplex *final,cuComplex *U_gpu,float *mascara, int N, int M)
{

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x+ threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;

    final[fila*M+(col)].x = U_gpu[fila*M+(col)].x * mascara[fila*M+(col)];
    final[fila*M+(col)].y = U_gpu[fila*M+(col)].y * mascara[fila*M+(col)];

} 
                   /*ESTA PARTE ES MERAMENTE PARA NORMALIZAR UNA MATRIZ */

                   
__global__ void Normalizar(float *U_gpu, int N, int M, float *minimo, float *maximo)
{
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    float mini=minimo[0];
    float maxi=maximo[0];
    U_gpu[(fila*M + col)] = (U_gpu[(fila*M + col)]-mini)/(maxi-mini); //Calculo la amplitud
}           
                   /*ESTA PARTE ES PARA LA RECONSTRUCCIÓN EN FASE, AMPLITUD O INSTENSIDAD*/
   
__global__ void Amplitud(float *U_gpu, cuComplex *dest_gpu, int N, int M)
{
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = sqrt(pow((dest_gpu[(fila*M+col)].x),2) + (pow((dest_gpu[(fila*M+col)].y),2)));  //Calculo la amplitud
}
                   
__global__ void Intensidad(float *U_gpu, cuComplex *dest_gpu, int N, int M)
{

	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = pow((dest_gpu[(fila*M+col)].x),2) + (pow((dest_gpu[(fila*M+col)].y),2));  //Calculo la intensidad
}

__global__ void Fase(float *U_gpu, cuComplex *dest_gpu, int N, int M){
                   
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = atan2f(dest_gpu[(fila*M+col)].y, dest_gpu[(fila*M+col)].x); //Calculo la fase
}
                   
                   /* Encontrar el minimo y máximo*/
__global__ void find_max_min(const float *U_gpu, float *max_val, float *min_val, int N, int M) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar datos en memoria compartida
    sdata[tid] = (i < N * M) ? U_gpu[i] : -INFINITY;  // Se utiliza -INFINITY como un valor inicial para asegurar que cualquier otro valor sea mayor
    __syncthreads();

    // Realizar reducción en memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Escribir resultados de reducción en el bloque de salida
    if (tid == 0) {
        max_val[blockIdx.x] = sdata[0];
        min_val[blockIdx.x] = sdata[blockDim.x];  // El mínimo se encuentra en la segunda mitad de sdata
    }
}

float find_max_min_gpu(const float *U_gpu, int N, int M, int block_size) {
    const int num_elements = N * M;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    float *max_vals_gpu, *min_vals_gpu;
    cudaMalloc((void**)&max_vals_gpu, num_blocks * sizeof(float));
    cudaMalloc((void**)&min_vals_gpu, num_blocks * sizeof(float));

    find_max_min<<<num_blocks, block_size, block_size * sizeof(float)>>>(U_gpu, max_vals_gpu, min_vals_gpu, N, M);

    float *max_vals_cpu = (float*)malloc(num_blocks * sizeof(float));
    float *min_vals_cpu = (float*)malloc(num_blocks * sizeof(float));

    cudaMemcpy(max_vals_cpu, max_vals_gpu, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_vals_cpu, min_vals_gpu, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float max_val = max_vals_cpu[0];
    float min_val = min_vals_cpu[0];

    // Encuentra el máximo y el mínimo globales
    for (int i = 1; i < num_blocks; ++i) {
        max_val = fmaxf(max_val, max_vals_cpu[i]);
        min_val = fminf(min_val, min_vals_cpu[i]);
    }

    free(max_vals_cpu);
    free(min_vals_cpu);
    cudaFree(max_vals_gpu);
    cudaFree(min_vals_gpu);

    return max_val,min_val;
}

""")
#Llamamos las funciones de CUDA

fase_refe = mod.get_function("fase_refe")
fft_shift = mod.get_function("fft_shift")
fft_shift_var_no_compleja = mod.get_function("fft_shift_var_no_compleja")
coordenadas_maximo= mod.get_function("coordenadas_maximo")
thresholding_kernel = mod.get_function("thresholding")
mascara_1er_cuadrante = mod.get_function("mascara_1er_cuadrante")
Normalizar = mod.get_function("Normalizar")
Reseteo = mod.get_function("reseteo")
Amplitud = mod.get_function("Amplitud")
Intensidad = mod.get_function("Intensidad")
Fase = mod.get_function("Fase")
find_max_min = mod.get_function("find_max_min")

#Creamos espacios de memoria en la GPU para trabajo 
holo = gpuarray.empty((N, M), np.complex128)
holo2 = gpuarray.empty((N, M), np.complex128)
temporal_gpu = gpuarray.empty((N, M), np.complex128)

# definición de espacios para trabajar
block_dim = (32, 32, 1)

# Mallado para la fft shift
grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)

#fft_shift
fft_shift_var_no_compleja(holo2,U_gpu,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)

#Fourier
plan = cu_fft.Plan((N,M), np.complex64, np.complex64)
cu_fft.fft(holo2, holo, plan)

#grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
#fft_shift
grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
fft_shift(holo2, holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
Amplitud(U_gpu,holo2,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)

#Obtención del espacio valor
grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
mascara_1er_cuadrante(holo,holo2,cuadrante_gpu, np.int32(N),np.int32(M), block=block_dim, grid=grid_dim)

#1280 x 960 las imagenes 
grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
Amplitud(U_gpu,holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)


grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
#Amplitud de la imagen hasta el momentojunpei girlfriend combatchidori co
fft_shift(holo2, holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

cu_fft.ifft(holo2, holo, plan)

grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
fft_shift(holo2, holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

block_dim = (32, 32, 1)
# Configuración de la cuadrícula y los bloques
block_size = 256
grid_size = 1  # Solo un bloque para buscar el máximo global

# Crear buffer para la posición del máximo en GPU
max_position_gpu = gpuarray.zeros((1,), dtype=np.int32)
# Crear buffer para la posición del máximo en GPU
max_position_gpu = gpuarray.zeros((U.shape[0],), dtype=np.int32)

# Ejecutar el kernel de búsqueda binaria
coordenadas_maximo(U_gpu, np.int32(U.shape[0]), np.int32(U.shape[1]), max_position_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

max_position_cpu = max_position_gpu.get()[0]
# Calcular las coordenadas (fila, columna) desde la posición
col_index, row_index = divmod(max_position_cpu, U.shape[1])

paso=0.2
fin=0
fy=col_index
fx=row_index
G_temp=G
suma_maxima=0
grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)


while fin==0:
    i=0
    j=0
    frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
    frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
    for j in range(len(frec_esp_y)): 
        for i in range(len(frec_esp_x)):
            fx_temp=frec_esp_x[i]
            fy_temp=frec_esp_y[j]
            theta_x=mt.asin((Fox - fx_temp) * lamb /(M*dx))
            theta_y=mt.asin((Foy - fy_temp) * lamb /(N*dy))
            #La propago
            fase_refe(holo2, holo, temporal_gpu, m_gpu, n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx_temp), np.float32(fy_temp), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)

            #La reconstruyo en fase

            Fase(U_gpu,holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
            
            #Ahora si toca encontrar maximos y minimos para normalizar
            
            max_value_gpu = gpuarray.max(U_gpu)
            min_value_gpu = gpuarray.min(U_gpu)
            
            #Normalizar
            
            Normalizar(U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
            
            #Aplicamos el thresholding
            
            thresholding_kernel(U_gpu,np.int32(N), np.int32(M), np.float32(threso), block=block_dim, grid=grid_dim)
            
            #Suma de la matriz

            sum_gpu = gpuarray.sum(U_gpu)
            
            temporal = sum_gpu.get()

            if(temporal>suma_maxima):
                x_max_out = fx_temp
                y_max_out = fy_temp
                suma_maxima = temporal
    G_temp = G_temp - 1
    
    if(x_max_out == fx):
        if(y_max_out ==fy):
            fin=1
    fx=x_max_out
    fy=y_max_out



fase_refe(holo2, holo, temporal_gpu, m_gpu, n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx), np.float32(fy), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)
max_value_gpu = gpuarray.max(U_gpu)
min_value_gpu = gpuarray.min(U_gpu)
                
#Normalizar
                
Normalizar(U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
    
#Obtención de la reconstrucción
mai = U_gpu.get()
finale = 255*mai.reshape((N, M))
l_temp=1
#nombre='Imagenes/reconstruccion/'+str(l_temp)+'.bmp'
#guardado(nombre, finale)

#Video es una matriz que guarda cada reconstrucción realizada
video = [finale]

#ELiminamos el primer archivo de la lista
archivos_ordenados.pop(0)
tiempo_final = time.time()
tiempo= tiempo_final-tiempo_inicial
tiempo_inicial= time.time()
tiempo_frame=[tiempo]
print(tiempo)
G = 1
paso=0.2
#Ahora la versión dinámica
for frame in archivos_ordenados:
    tiempo_inicial= time.time()
    archivo = carpeta2+str(frame)

    #Esto es lo que dijo carlos yo creo
    replica = lectura(archivo)
    #Sera que esta se puede paralelizar?

    replica=ajuste_tamano(replica)
    U = asarray(replica)
    U = U.astype(np.float32)
    U_gpu = gpuarray.to_gpu(U)
    grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
    block_dim = (32, 32, 1)
    #fft_shift
    fft_shift_var_no_compleja(holo2,U_gpu,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)

    #Fourier
    cu_fft.fft(holo2, holo, plan)
    
    #fft_shift
    grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
    fft_shift(holo2, holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

    #Obtención del espacio valor
    grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
    mascara_1er_cuadrante(holo,holo2,cuadrante_gpu, np.int32(N),np.int32(M), block=block_dim, grid=grid_dim)

    grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
    #Amplitud de la imagen hasta el momentojunpei girlfriend combatchidori co
    fft_shift(holo2, holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

    cu_fft.ifft(holo2, holo, plan)

    grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
    fft_shift(holo2, holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

    G_temp=G
    suma_maxima=0
    grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)


    while fin==0:
        i=0
        j=0
        frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
        frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
        for j in range(len(frec_esp_y)):
            for i in range(len(frec_esp_x)):
                fx_temp=frec_esp_x[i]
                fy_temp=frec_esp_y[j]
                theta_x=mt.asin((Fox - fx_temp) * lamb /(M*dx))
                theta_y=mt.asin((Foy - fy_temp) * lamb /(N*dy))
                #La propago
                #Revisar función por función, para ver los tiempos
                fase_refe(holo2, holo, temporal_gpu, m_gpu, n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx_temp), np.float32(fy_temp), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)

                #La reconstruyo en fase
                
                Fase(U_gpu,holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
                
                #Ahora si toca encontrar maximos y minimos para normalizar
                
                max_value_gpu = gpuarray.max(U_gpu)
                min_value_gpu = gpuarray.min(U_gpu)
                
                #Normalizar
                #Revisar función por función, para ver los tiempos
                Normalizar(U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
                
                #Aplicamos el thresholding
                
                thresholding_kernel(U_gpu,np.int32(N), np.int32(M), np.float32(threso), block=block_dim, grid=grid_dim)
                
                #Suma de la matriz

                sum_gpu = gpuarray.sum(U_gpu)
                
                temporal = sum_gpu.get()

                if(temporal>suma_maxima):
                    x_max_out = fx_temp
                    y_max_out = fy_temp
                    suma_maxima = temporal
        G_temp = G_temp - 1
        
        if(x_max_out == fx):
            if(y_max_out ==fy):
                fin=1
        fx=x_max_out
        fy=y_max_out



    fase_refe(holo2, holo, temporal_gpu, m_gpu, n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx), np.float32(fy), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)
    Fase(U_gpu,holo, np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
    max_value_gpu = gpuarray.max(U_gpu)
    min_value_gpu = gpuarray.min(U_gpu)
                
    #Normalizar
                
    Normalizar(U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
    
    #Obtención de la reconstrucción
    mai = U_gpu.get()
    finale = 255*mai.reshape((N, M))
    video.append(finale)
    tiempo_final = time.time()
    tiempo = tiempo_final-tiempo_inicial
    tiempo_frame.append(tiempo)
    l_temp=l_temp+1
    #nombre='Imagenes/reconstruccion/'+str(l_temp)+'.bmp'
    #guardado(nombre, finale)
# Suponiendo que 'video' es tu array con las reconstrucciones
video_array = np.array(video)

# Número total de frames
num_frames = video_array.shape[0]
print("Ya acabo de procesar, falta el plt")
# Configuración inicial de la figura
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Añadir un eje para el slider
ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03])
slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=1, valstep=1)

# Función de actualización específica para el slider
def update(val):
    frame = int(slider.val)
    ax.clear()
    ax.imshow(video_array[frame])
    ax.set_title(f'Frame {frame}')
    plt.draw()

# Conectar la función de actualización al evento del slider
slider.on_changed(update)

# Mostrar la figura
#plt.show()

# Guardar el array en un archivo de texto
np.savetxt('tiempos.txt', tiempo_frame, fmt='%f', delimiter='\t')

video=np.array(video)
num_frames, height, width = video.shape

# Define el formato del video y crea un objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes ajustar el códec según tus necesidades
video_path = 'video_salida.mp4'  # Puedes cambiar la extensión según el formato deseado (mp4, gif, etc.)
imageio.mimsave(video_path, video, fps=30)
