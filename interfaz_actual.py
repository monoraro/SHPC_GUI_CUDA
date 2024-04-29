import cv2
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
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
import time
import uuid
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

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

class CameraApp:
    def __init__(self, ventana):
        #Llamamos funciones de cuda, creamos la ventana de la app e inicializamos
        self.llamado_funciones_cuda()
        self.ventana = ventana
        self.ini=0

        #Parametros de la venatana
        ventana.geometry("1920x1080")
        ventana.title("Captura y Transformación de Cámara")

        #Llamamos opencv para leer la cámara
        self.cap = cv2.VideoCapture(0)

        # Establecer el tema "cosmo" al inicio (el estilo blanquito)
        ventana.style = ttkb.Style(theme="cosmo")

        # Encabezado con línea superior completa
        self.encabezado_frame = ttk.Frame(ventana, padding="10")
        self.encabezado_frame.grid(row=0, column=0, columnspan=8, sticky="ew")
        self.titulo_label = ttk.Label(self.encabezado_frame, text="Aplicación de Reconstrucción", font=("Helvetica", 14, "bold"))
        self.titulo_label.grid(row=0, column=0, sticky="w")
        self.hline = ttk.Separator(ventana, orient="horizontal")
        self.hline.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(5, 10))
        self.ventana.columnconfigure(0, weight=1)  # Expandir la primera columna

        # Selector de tema en la esquina derecha superior
        ttk.Label(ventana, text="Seleccionar tema:").grid(row=0, column=8, padx=(100, 5), sticky="e")
        self.theme_selector = ttk.Combobox(ventana, values=["cosmo", "darkly", "vapor", "cyborg"], state="readonly",width= 0)
        self.theme_selector.grid(row=0, column=9, padx=(0, 20), sticky="ew")
        self.theme_selector.set(ventana.style.theme.name)  # Establecer el tema actual como seleccionado
        self.theme_selector.bind("<<ComboboxSelected>>", self.cambiar_tema)

        # Configuración de los labels para mostrar las imágenes
        self.label_original = ttk.Label(ventana)
        self.label_original.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(0, 10))

        # Cuadro de texto uno (dx)
        ttk.Label(ventana, text="Definición de parámetros", font=("Helvetica", 14, "bold")).grid(row=7, column=1, padx=(0, 5), pady=10,columnspan=3)
        ttk.Label(ventana, text="Delta x:").grid(row=8, column=1, padx=(10, 5), pady=10,sticky="e")
        self.entry_param1 = tk.Entry(ventana,width= 5)
        self.entry_param1.grid(row=8, column=2, padx=(5, 0), pady=10, sticky="w")

        # Insertar el valor inicial de dx
        self.entry_param1.insert(0, "3.75")

        #Cuadro de texto dos (dy)
        ttk.Label(ventana, text="Delta y:").grid(row=9, column=1, padx=(10, 5), pady=10, sticky="e")
        self.entry_param2 = tk.Entry(ventana,width= 5)
        self.entry_param2.grid(row=9, column=2, padx=(5, 0), pady=10, sticky="w")

        # Insertar el valor inicial de dy
        self.entry_param2.insert(0, "3.75")

        # Combobox con valores 1, 2, 3 y 4
        ttk.Label(ventana, text="Cuadrante:").grid(row=8, column=2, padx=(10, 5), pady=10, sticky="e")
        self.entry_param4 = ttk.Combobox(root, values=[1, 2, 3, 4], state="readonly", width=1)
        self.entry_param4.grid(row=8, column=3, padx=(0, 0), pady=10, sticky="w")

        #Cuadro de texto tres (lambda)
        ttk.Label(ventana, text="Longitud de onda:").grid(row=9, column=2, padx=(10, 5), pady=10, sticky="e")
        self.entry_param3 = tk.Entry(ventana,width= 5)
        self.entry_param3.grid(row=9, column=3, padx=(0, 0), pady=10, sticky="w")

        # Insertar el valor inicial de longitud de onda
        self.entry_param3.insert(0, "0.633")

        # Boton de aplicar la configuración
        self.boton_aplicar = ttk.Button(ventana, text="Aplicar", command=self.aplicar_transformaciones)
        self.boton_aplicar.grid(row=8, column= 3, padx=10, pady=10, rowspan=2, sticky='e')

        #Titulo de grabar
        ttk.Label(ventana, text="Grabar", font=("Helvetica", 12, "bold")).grid(row=7, column=5, padx=(0, 0), pady=10,columnspan=2)
        
        ttk.Label(ventana, text="¿Qué deseas guardar?").grid(row=8, column=5, padx=(10, 5), pady=10, sticky="e")
        self.tipo_grabar = ttk.Combobox(root, values=["Solo la reconstrucción", "Holagrama y reconstrucción"], state="readonly", width=20)
        self.tipo_grabar.grid(row=8, column=6, padx=(0, 0), pady=10, sticky="w")

        # Boton de grabar (no funcional)
        self.boton_grabar = ttk.Button(ventana, text="Grabar", command=self.toggle_grabacion)
        self.boton_grabar.grid(row=9, column=5, padx=10, pady=10,columnspan=2)
        self.boton_grabar.config(state="disabled")  # Deshabilita el botón si no hay texto

        # Imágenes de fondo negro
        self.black_image1 = Image.new("RGB", (500, 500), "black")
        self.black_image2 = Image.new("RGB", (500, 500), "black")
        self.black_image3 = Image.new("RGB", (500, 500), "black")

        self.black_image1_tk = ImageTk.PhotoImage(self.black_image1)
        self.black_image2_tk = ImageTk.PhotoImage(self.black_image2)
        self.black_image3_tk = ImageTk.PhotoImage(self.black_image3)
        
        ttk.Label(ventana, text="Holograma", font=("Helvetica", 12, "bold")).grid(row=2, column=1, padx=(0, 0), pady=10,columnspan=3)
        self.label_original = ttk.Label(ventana, image=self.black_image1_tk)
        self.label_original.grid(row=3, column=1, columnspan=3,rowspan=3)

        ttk.Label(ventana, text="Transformada de fourier", font=("Helvetica", 12, "bold")).grid(row=2, column=4, padx=(0, 0), pady=10,columnspan=3)
        self.label_transformacion1 = ttk.Label(ventana, image=self.black_image2_tk)
        self.label_transformacion1.grid(row=3, column=4, columnspan=3,rowspan=3)

        ttk.Label(ventana, text="Reconstrucción en fase", font=("Helvetica", 12, "bold")).grid(row=2, column=7, padx=(0, 0), pady=10,columnspan=3)
        self.label_transformacion2 = ttk.Label(ventana, image=self.black_image3_tk)
        self.label_transformacion2.grid(row=3, column=7, columnspan=3,rowspan=3)

        # Crear un botón que abrirá la nueva ventana
        self.btn_abrir_ventana = tk.Button(ventana, text="Abrir Nueva Ventana", command=self.abrir_nueva_ventana)
        self.btn_abrir_ventana.grid(row=7, column=8)

        self.hline = ttk.Separator(ventana, orient="horizontal")
        self.hline.grid(row=6, column=0, columnspan=10, sticky="ew", pady=(10, 0))

        self.hline = ttk.Separator(ventana, orient="vertical")
        self.hline.grid(row=6, column=4, rowspan=10, sticky="wns", pady=(10, 10))
        
        # Llamar al método para capturar y mostrar fotogramas
        self.mostrar_fotogramas()
        self.ini2=0
        self.grabacion = 0
        self.tipo = "a"
    #Esta función lee las funciones implementadas en el sourcemodule o kernel 
    def llamado_funciones_cuda(self):
        
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

    def cambiar_tema(self, event):
        nuevo_tema = self.theme_selector.get()
        self.ventana.style.theme_use(nuevo_tema)

    def mostrar_fotogramas(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_original.imgtk = imgtk
            self.label_original.configure(image=imgtk)
        self.ventana.after(20, self.mostrar_fotogramas)

    
    def toggle_grabacion(self):
        if(self.grabacion==0):
            self.tipo_grabar.config(state="disabled")
            self.boton_grabar.config(text="Detener grabacion")
            self.grabacion=1
            self.tipo = self.tipo_grabar.get()
            self.recons = []
            self.origen = []
        else:
            
            self.boton_grabar.config(text="Cargando archivos")
            #Si entra y ya grabó, necesita guardar, por lo tanto
            self.grabacion=0
            self.boton_grabar.config(state="disabled") 
            if(self.tipo=="Holagrama y reconstrucción"):
                
                video_origen = f'origen_{uuid.uuid4()}.mp4'
                video_recons = f'reconstruccion_{uuid.uuid4()}.mp4'
                fps = 24
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = self.origen[0].shape[1]
                frame_height = self.origen[0].shape[0]
                video = cv2.VideoWriter(video_origen, fourcc, fps, (frame_width, frame_height), isColor=False)
                frame_width = self.recons[0].shape[1]
                frame_height = self.recons[0].shape[0]
                video2 = cv2.VideoWriter(video_recons, fourcc, fps, (frame_width, frame_height), isColor=False)
                for i in range(len(self.origen)):
                    frame = self.origen[i]
                    video.write(frame)
                    frame = self.recons[i]
                    video2.write(frame)
                video.release()
                video2.release()
                cv2.destroyAllWindows()
            elif(self.tipo=="Solo la reconstrucción"):
                video_recons = f'reconstruccion_{uuid.uuid4()}.mp4'
                fps = 24
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = self.recons[0].shape[1]
                frame_height = self.recons[0].shape[0]
                video2 = cv2.VideoWriter(video_recons, fourcc, fps, (frame_width, frame_height), isColor=False)
                for i in range(len(self.recons)):
                    frame = self.recons[i]
                    video2.write(frame)
                print(self.recons[0])
                video2.release()
                cv2.destroyAllWindows()
            self.boton_grabar.config(text="Guardar")
            self.boton_grabar.config(state="normal") 
            self.tipo_grabar.config(state="readonly")
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
        print("entro a transforamción 1")
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
        self.start_time = time.time()
        self.frame_count = 0
        self.boton_grabar.config(state="normal")  # Habilita el botón
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
        if(self.tipo=="Holagrama y reconstrucción"):
            self.origen.append(frame)
            self.recons.append(finale.astype(np.uint8))
            
        elif(self.tipo=="Solo la reconstrucción"):
            self.recons.append(finale.astype(np.uint8))
        self.frame1 = frame1.astype(np.uint8)
        return frame1,finale
    
    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Actualiza cada segundo
            fps = self.frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.start_time = time.time()

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
                
                img_original = img_original.resize((round(N*0.75),round(M*0.75)))
                #transformada de fourier
                frame_trans1 = frame_trans1.astype(np.uint8)
                
                img_trans1 = Image.fromarray(cv2.cvtColor(frame_trans1, cv2.COLOR_GRAY2RGB))
                img_trans1 = img_trans1.resize((round(N*0.75),round(M*0.75)))
                
                frame_trans2 = frame_trans2.astype(np.uint8)
                img_trans2 = Image.fromarray(cv2.cvtColor(frame_trans2, cv2.COLOR_GRAY2RGB))
                img_trans2 = img_trans2.resize((round(N*0.75),round(M*0.75)))
                
                #Resultado
                img_original_tk = ImageTk.PhotoImage(image=img_original)
                img_trans1_tk = ImageTk.PhotoImage(image=img_trans1)
                img_trans2_tk = ImageTk.PhotoImage(image= img_trans2)

                # Mostrar las imágenes en los widgets Label
                self.label_original.configure(image=img_original_tk)
                self.label_original.image = img_original_tk

                self.label_transformacion1.configure(image=img_trans1_tk)
                self.label_transformacion1.image = img_trans1_tk

                self.label_transformacion2.configure(image=img_trans2_tk)
                self.label_transformacion2.image = img_trans2_tk
                self.update()

        # Llamar al método cada 20 milisegundos para mostrar el siguiente fotograma
        self.ventana.after(40, self.mostrar_fotogramas)
    
    #Esta función permite reconocer los parámetros para la reconstrucción
    def aplicar_transformaciones(self):
        # Obtener los parámetros ingresados por el usuario
        self.dx = float(self.entry_param1.get())
        self.dy = float(self.entry_param2.get())
        self.lamb = float(self.entry_param3.get())
        self.cuadrante = float(self.entry_param4.get())
        self.ini=1
        # Aquí puedes realizar alguna acción con los parámetros, como aplicar alguna transformación adicional
    def abrir_nueva_ventana(self):
        # Crear una nueva ventana
        self.nueva_ventana = tk.Toplevel(self.ventana)
        self.nueva_ventana.title("Plt con tkinter")
       
        # Configurar la figura de Matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Imagen de la cámara en tiempo real")
        self.ax.axis('off')
        self.ax.autoscale(True)
        self.img1 = self.ax.imshow(cv2.cvtColor(self.frame1, cv2.COLOR_GRAY2RGB))
        # Actualizar la imagen en las figuras utilizando FuncAnimation
        ani = FuncAnimation(self.fig, self.actualizar)
        # Agregar el lienzo de Matplotlib a la ventana de Tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=self.nueva_ventana)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Agregar la barra de herramientas de Matplotlib
        toolbar = NavigationToolbar2Tk(canvas, self.nueva_ventana, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
                
        self.nueva_ventana.mainloop()

    def actualizar(self,frame):
        self.img1.set_array(cv2.cvtColor(self.frame1, cv2.COLOR_GRAY2RGB))
        agua = self.img1
        return agua,
        
# Creación de la ventana y de la aplicación
root = tk.Tk()  # Utilizamos una ventana normal de Tkinter
app = CameraApp(root)
root.mainloop()
