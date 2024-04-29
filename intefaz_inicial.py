import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

def actualizar_imagen(frame):
    # Capturar un cuadro de la cámara
    ret, frame = captura.read()
    if ret:
        # Mostrar la imagen
        img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img,

# Inicializar la captura de la cámara
captura = cv2.VideoCapture(0)  # 0 representa la cámara predeterminada, puedes cambiar el número si tienes varias cámaras

# Configurar la figura de Matplotlib
fig, ax = plt.subplots()
ax.set_title("Imagen de la cámara en tiempo real")
ax.axis('off')
img = ax.imshow(cv2.cvtColor(captura.read()[1], cv2.COLOR_BGR2RGB))

# Permitir zoom en la figura
ax.set_xlim(0, img.get_array().shape[1])
ax.set_ylim(img.get_array().shape[0], 0)
ax.autoscale(False)

# Actualizar la imagen en la figura utilizando FuncAnimation
ani = FuncAnimation(fig, actualizar_imagen, interval=50, blit=True)

# Crear la ventana de Tkinter
root = tk.Tk()
root.title("Transmisión de la cámara en tiempo real")

# Agregar el lienzo de Matplotlib a la ventana de Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Agregar la barra de herramientas de Matplotlib
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
toolbar.pack(side=tk.BOTTOM, fill=tk.X)

# Liberar la captura al cerrar la ventana
def cerrar_ventana(event):
    if event.keysym == 'q':
        captura.release()
        root.destroy()

root.bind('<Key>', cerrar_ventana)  # Detectar la tecla 'q' para cerrar la ventana

# Iniciar el bucle principal de Tkinter
root.mainloop()
