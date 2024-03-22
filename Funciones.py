import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import glob
##Funciones hechas dentro de CUDA
#Funciones desde python

#Función de lectura de una imagen dada
def lectura(name_file):
    replica = Image.open("./" +
                         str(name_file)).convert('L')
    replica.save("./Imagenes/copia.png")
    return (replica)
def ajuste_tamano(archivo):
    N_, M_ = archivo.size
    N_ = N_ // 64
    M_ = M_ // 64
    replica = np.resize(archivo, (N_*64,M_*64))
    return (replica)

def ajuste_tamano1(archivo):
    N_, M_ = archivo.shape
    N_ = N_ // 64
    M_ = M_ // 64
    vector = archivo.flatten()
    replica = np.resize(archivo, (N_*64,M_*64))
    return (replica)

def lectura_continua(direccion):
    cv_img = []
    arepa=glob.glob(direccion)
    archivos = sorted(arepa, key=lambda x: x, reverse=True)
    for img in archivos:
        print(img)
        n = Image.open(img).convert('L')
        cv_img.append(n)
    return (cv_img)

# Función para el guardado de la imagen
def guardado(name_out, matriz):
    
    resultado = Image.fromarray(matriz)
    resultado = resultado.convert('RGB')
    resultado.save("./"+str(name_out))

# Función para graficar y ponerle nombre a los ejes
def mostrar(matriz, titulo, ejex, ejey):
    plt.imshow(matriz, cmap='gray')
    plt.title(str(titulo))
    plt.xlabel(str(ejex))
    plt.ylabel(str(ejey))
    plt.show()

# Calculo de la magnitud, pero hagamos esto en cuda
def amplitud(matriz):
    amplitud = np.abs(matriz)
    return (amplitud)



def intensidad(matriz):
    intensidad = np.abs(matriz)
    intensidad = np.power(intensidad, 2)
    return (intensidad)


def fase(matriz):
    fase = np.angle(matriz, deg=False)
    return (fase)


def dual_img(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.show()


def dual_save(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.savefig('./Imagenes/guardado.png', dpi=1000)
