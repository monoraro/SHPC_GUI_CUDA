import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule(
"""

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
"""
)

