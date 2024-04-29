
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from numpy import asarray
import math as mt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from pycuda.reduction import ReductionKernel
from Funciones import *

#Carga del kernel de cuda

mod = SourceModule("""
#include <math.h>
#include <stdio.h>
#include <cuComplex.h>
#include <limits.h>

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
                   
__global__ void Amplitud_log(float *U_gpu, cuComplex *dest_gpu, int N, int M)
{
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = log(sqrt(pow((dest_gpu[(fila*M+col)].x),2) + (pow((dest_gpu[(fila*M+col)].y),2))));  //Calculo la amplitud
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



#Todo sea para la GUI
class AplicacionCamara:
    #Creamos el entorno de la GUI
    def __init__(self, ventana):
        
        #Captura de funcione
        self.fase_refe = mod.get_function("fase_refe")
        self.fft_shift = mod.get_function("fft_shift")
        self.fft_shift_var_no_compleja = mod.get_function("fft_shift_var_no_compleja")
        self.coordenadas_maximo= mod.get_function("coordenadas_maximo")
        self.thresholding_kernel = mod.get_function("thresholding")
        self.mascara_1er_cuadrante = mod.get_function("mascara_1er_cuadrante")
        self.Normalizar = mod.get_function("Normalizar")
        self.Reseteo = mod.get_function("reseteo")
        self.Amplitud = mod.get_function("Amplitud")
        self.Intensidad = mod.get_function("Intensidad")
        self.Fase = mod.get_function("Fase")
        self.find_max_min = mod.get_function("find_max_min")
        self.logaritmo = mod.get_function("Amplitud_log")
        self.ini=0
        self.ventana = ventana
        self.ventana.title("Captura y Transformación de Cámara")

        # Inicializar la captura de la cámara
        self.cap = cv2.VideoCapture(0)
        
        # Crear widgets para mostrar las imágenes y títulos
        self.label_original = tk.Label(ventana)
        self.label_original.grid(row=0, column=0, sticky="nsew")
        self.label_transformacion1 = tk.Label(ventana)
        self.label_transformacion1.grid(row=0, column=2, sticky="nsew")
        self.label_transformacion2 = tk.Label(ventana)
        self.label_transformacion2.grid(row=0, column=4, sticky="nsew")

        # Crear widgets para las cajas de entrada y botón
        self.title = tk.Label(ventana, text="Original")
        self.title.grid(row=1, column=0, sticky="ew")
        self.title = tk.Label(ventana, text="Transformada de fourier")
        self.title.grid(row=1, column=2, sticky="ew")
        self.title = tk.Label(ventana, text="Mapa de fase")
        self.title.grid(row=1, column=4, sticky="ew")

        self.label_param1 = tk.Label(ventana, text="dx")
        self.label_param1.grid(row=2, column=0, sticky="e")
        self.entry_param1 = tk.Entry(ventana)
        self.entry_param1.grid(row=2, column=1, sticky="ew")
        
        self.label_param2 = tk.Label(ventana, text="dy:")
        self.label_param2.grid(row=3, column=0, sticky="e")
        self.entry_param2 = tk.Entry(ventana)
        self.entry_param2.grid(row=3, column=1, sticky="ew")
        
        self.label_param3 = tk.Label(ventana, text="longitud de onda:")
        self.label_param3.grid(row=3, column=2, sticky="e")
        self.entry_param3 = tk.Entry(ventana)
        self.entry_param3.grid(row=3, column=3, sticky="ew")

        self.label_param4 = tk.Label(ventana, text="cuadrante")
        self.label_param4.grid(row=2, column=2, sticky="e")
        self.entry_param4 = tk.Entry(ventana)
        self.entry_param4.grid(row=2, column=3, sticky="ew", pady=20)
        
        self.boton_aplicar = tk.Button(ventana, text="Aplicar", command=self.aplicar_transformaciones)
        self.boton_aplicar.grid(row=4, column=2)

        # Llamar al método para capturar y mostrar fotogramas
        self.mostrar_fotogramas()
        self.ini2=0
    #Esta función lo que hace es tomar el fotograma de la camara del pc
    #Toca adaptarla para la camara que se planea usar
    def capturar_fotograma(self):
        # Capturar un fotograma de la cámara
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        else: 
            return None
    
    #Esta transforamción es para el 1er frame
    def transformacion_1(self, frame):
        frame=ajuste_tamano1(frame)
        N, M = frame.shape
        
        self.N = N
        self.M = M
        x = np.arange(0, M, 1)
        y = np.arange(0, N, 1)

        #Un meshgrid para la paralelizacion
        m, n = np.meshgrid(x - (M/2), y - (N/2))
        G=4
        k = 2*mt.pi/self.lamb
        Fox = M/2
        Foy = N/2
        threso = 0.2
        #Esta variable sirve para inicializar el valor mínimo de la suma
        suma_max = np.array([[0]]) 
        #Definicion de tipos de variables compatibles con cuda
        U = frame
        U = U.astype(np.float32)
        
        m = m.astype(np.float32)
        n = n.astype(np.float32)
        #Variables definidas a la gpu
        self.U_gpu = gpuarray.to_gpu(U)
        if(self.ini2==0):
            self.m_gpu = gpuarray.to_gpu(m)
            self.n_gpu = gpuarray.to_gpu(n)

        if(int(self.cuadrante==1)):
            primer_cuadrante= np.zeros((N,M))
            primer_cuadrante[0:round(N/2 - (N*0.1)),round(M/2 + (M*0.1)):M] = 1
            primer_cuadrante = primer_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(primer_cuadrante)
        if(int(self.cuadrante==2)):
            segundo_cuadrante= np.zeros((N,M))
            segundo_cuadrante[0:round(N/2 -(N*0.1)),0:round(M/2 - (M*0.1))] = 1
            segundo_cuadrante = segundo_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(segundo_cuadrante)

        if(int(self.cuadrante)==3):
            tercer_cuadrante= np.zeros((N,M))
            tercer_cuadrante[round(N/2 +(N*0.1)):N,0:round(M/2 - (M*0.1))] = 1
            tercer_cuadrante = tercer_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(tercer_cuadrante)

        if(int(self.cuadrante)==4):
            cuarto_cuadrante= np.zeros((N,M))
            cuarto_cuadrante[round(N/2 +(N*0.1)):N,round(M/2 + (M*0.1)):M] = 1
            cuarto_cuadrante = cuarto_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(cuarto_cuadrante)

        #Creamos espacios de memoria en la GPU para trabajo
        if(self.ini2==0):
            self.holo = gpuarray.empty((N, M), np.complex128)
            self.holo2 = gpuarray.empty((N, M), np.complex128)
            self.temporal_gpu = gpuarray.empty((N, M), np.complex128)

        # definición de espacios para trabajar
        block_dim = (32, 32, 1)

        # Mallado para la fft shift
        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        #fft_shift
        self.fft_shift_var_no_compleja(self.holo2,self.U_gpu,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)

        #Fourier
        if(self.ini2==0):
            self.plan = cu_fft.Plan((N,M), np.complex64, np.complex64)
            self.ini2 = 1
        cu_fft.fft(self.holo2, self.holo, self.plan)

        #grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        #fft_shift
        print("maguiver")
        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
        
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.logaritmo(self.U_gpu,self.holo2,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
        max_value_gpu = gpuarray.max(self.U_gpu)
        min_value_gpu = gpuarray.min(self.U_gpu)

        #Normalizar
        self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
        
        
        mai = self.U_gpu.get()
        frame1 = 255*mai.reshape((N, M))
        #Obtención del espacio valor
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.mascara_1er_cuadrante(self.holo,self.holo2,self.cuadrante_gpu, np.int32(N),np.int32(M), block=block_dim, grid=grid_dim)

        #1280 x 960 las imagenes 
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.Amplitud(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)


        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        #Amplitud de la imagen hasta el momentojunpei girlfriend combatchidori co
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

        cu_fft.ifft(self.holo2, self.holo, self.plan)

        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

        block_dim = (32, 32, 1)
        # Configuración de la cuadrícula y los bloques
        block_size = 256
        grid_size = 1  # Solo un bloque para buscar el máximo global

        # Crear buffer para la posición del máximo en GPU
        self.max_position_gpu= gpuarray.zeros((U.shape[0],), dtype=np.int32)

        # Ejecutar el kernel de búsqueda binaria
        self.coordenadas_maximo(self.U_gpu, np.int32(U.shape[0]), np.int32(U.shape[1]), self.max_position_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

        max_position_cpu = self.max_position_gpu.get()[0]
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

                    #La propago
                    self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx_temp), np.float32(fy_temp), np.float32(self.lamb), np.float32(self.dx), np.float32(self.dy), block=block_dim, grid=grid_dim)

                    #La reconstruyo en fase

                    self.Fase(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
                    
                    #Ahora si toca encontrar maximos y minimos para normalizar
                    
                    max_value_gpu = gpuarray.max(self.U_gpu)
                    min_value_gpu = gpuarray.min(self.U_gpu)
                    
                    #Normalizar
                    
                    self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
                    
                    #Aplicamos el thresholding
                    
                    self.thresholding_kernel(self.U_gpu,np.int32(N), np.int32(M), np.float32(threso), block=block_dim, grid=grid_dim)
                    
                    #Suma de la matriz

                    self.sum_gpu = gpuarray.sum(self.U_gpu)
                    
                    temporal = self.sum_gpu.get()

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


        self.fx = fx
        self.fy = fy
        self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx), np.float32(fy), np.float32(self.lamb), np.float32(self.dx), np.float32(self.dy), block=block_dim, grid=grid_dim)
        max_value_gpu = gpuarray.max(self.U_gpu)
        min_value_gpu = gpuarray.min(self.U_gpu)
                        
        #Normalizar
                        
        self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
            
        #Obtención de la reconstrucción
        mai = self.U_gpu.get()
        frame = 255*mai.reshape((N, M))
        self.ini = 2
        return frame1,frame
    
    #Algoritmo para los demás frames
    def transformacion_2(self, frame):
        frame=ajuste_tamano1(frame)
        G = 1
        paso=0.2
        k = 2*mt.pi/self.lamb
        U = asarray(frame)
        U = U.astype(np.float32)
        U_gpu = gpuarray.to_gpu(U)
        N = self.N
        M = self.M
        Fox = M/2
        Foy = N/2
        block_dim = (32, 32, 1)
        grid_dim = (self.N // (block_dim[0]*2), self.M // (block_dim[1]*2), 1)
        
        #fft_shift
        self.fft_shift_var_no_compleja(self.holo2,U_gpu,np.int32(self.N),np.int32(self.M),block=block_dim, grid=grid_dim)

        #Fourier
        cu_fft.fft(self.holo2, self.holo, self.plan)
        
        #fft_shift
        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        self.fft_shift(self.holo2, self.holo, np.int32(self.N), np.int32(self.M), block=block_dim, grid=grid_dim)
        
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.logaritmo(self.U_gpu,self.holo2,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
        max_value_gpu = gpuarray.max(self.U_gpu)
        min_value_gpu = gpuarray.min(self.U_gpu)

        #Normalizar
        self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
        
        
        mai = self.U_gpu.get()
        frame1 = 255*mai.reshape((N, M))
        #Obtención del espacio valor
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.mascara_1er_cuadrante(self.holo,self.holo2,self.cuadrante_gpu, np.int32(N),np.int32(M), block=block_dim, grid=grid_dim)

        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        #Amplitud de la imagen hasta el momentojunpei girlfriend combatchidori co
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

        cu_fft.ifft(self.holo2, self.holo, self.plan)

        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

        G_temp=G
        suma_maxima=0
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        fx = self.fx
        fy = self.fy
        fin=0
        while fin==0:

            frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
            frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
            for j in range(len(frec_esp_y)):
                for i in range(len(frec_esp_x)):
                    fx_temp=frec_esp_x[i]
                    fy_temp=frec_esp_y[j]
                    #La propago
                    #Revisar función por función, para ver los tiempos
                    self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx_temp), np.float32(fy_temp), np.float32(self.lamb), np.float32(self.dx), np.float32(self.dy), block=block_dim, grid=grid_dim)

                    #La reconstruyo en fase
                    self.Fase(U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
                    
                    #Ahora si toca encontrar maximos y minimos para normalizar
                    
                    max_value_gpu = gpuarray.max(U_gpu)
                    min_value_gpu = gpuarray.min(U_gpu)
                    
                    #Normalizar
                    #Revisar función por función, para ver los tiempos
                    self.Normalizar(U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
                    
                    #Aplicamos el thresholding
                    self.thresholding_kernel(U_gpu,np.int32(N), np.int32(M), np.float32(0.2), block=block_dim, grid=grid_dim)
                    
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
        
        self.fx = fx
        self.fy = fy
        self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx), np.float32(fy), np.float32(self.lamb), np.float32(self.dx), np.float32(self.dy), block=block_dim, grid=grid_dim)
        self.Fase(U_gpu,self.holo, np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
        max_value_gpu = gpuarray.max(U_gpu)
        min_value_gpu = gpuarray.min(U_gpu)
                    
        #Normalizar
                    
        self.Normalizar(U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
        
        #Obtención de la reconstrucción
        mai = U_gpu.get()
        finale = 255*mai.reshape((N, M))
        
        return frame1,finale

    #Esta es la función principal de la GUI, se encarga de actualizar fotograma a fotograma
    def mostrar_fotogramas(self):
        # Capturar un fotograma
        frame = self.capturar_fotograma()

        if frame is not None:
            if self.ini != 0:
                # Aplicar las transformaciones
                if self.ini == 1:
                    frame_trans1,frame_trans2= self.transformacion_1(frame)
                else:
                    frame_trans1,frame_trans2= self.transformacion_2(frame)
                # Convertir los fotogramas a formato adecuado para Tkinter
                M,N = frame.shape
                img_original = Image.fromarray((cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)))
                img_original = img_original.resize((round(N*0.5),round(M*0.5)))
                #transformada de fourier
                frame_trans1 = frame_trans1.astype(np.uint8)

                img_trans1 = Image.fromarray(cv2.cvtColor(frame_trans1, cv2.COLOR_GRAY2RGB))
                img_trans1 = img_trans1.resize((round(N*0.5),round(M*0.5)))
                frame_trans2 = frame_trans2.astype(np.uint8)
                img_trans2 = Image.fromarray(cv2.cvtColor(frame_trans2, cv2.COLOR_GRAY2RGB))
                img_trans2 = img_trans2.resize((round(N*0.5),round(M*0.5)))
                #Resultado
                img_original_tk = ImageTk.PhotoImage(image=img_original)
                img_trans1_tk = ImageTk.PhotoImage(image=img_trans1)
                img_trans2_tk = ImageTk.PhotoImage(image=img_trans2)

                # Mostrar las imágenes en los widgets Label
                self.label_original.configure(image=img_original_tk)
                self.label_original.image = img_original_tk

                self.label_transformacion1.configure(image=img_trans1_tk)
                self.label_transformacion1.image = img_trans1_tk

                self.label_transformacion2.configure(image=img_trans2_tk)
                self.label_transformacion2.image = img_trans2_tk

        # Llamar al método cada 20 milisegundos para mostrar el siguiente fotograma
        self.ventana.after(2, self.mostrar_fotogramas)
    
    #Esta función permite reconocer los parámetros para la reconstrucción
    def aplicar_transformaciones(self):
        # Obtener los parámetros ingresados por el usuario
        self.dx = float(self.entry_param1.get())
        self.dy = float(self.entry_param2.get())
        self.lamb = float(self.entry_param3.get())
        self.cuadrante = float(self.entry_param4.get())
        self.ini=1
        # Aquí puedes realizar alguna acción con los parámetros, como aplicar alguna transformación adicional


if __name__ == "__main__":
    # Crear la ventana de la aplicación
    ventana = tk.Tk()
    
    # Crear la aplicación y ejecutarla
    app = AplicacionCamara(ventana)

    # Ejecutar el bucle principal de la aplicación
    ventana.mainloop()

    # Liberar la captura de la cámara al finalizar
    app.cap.release()