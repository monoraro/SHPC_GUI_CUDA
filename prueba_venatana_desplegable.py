import tkinter as tk
from tkinter import ttk
import time

class MiApp:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Ejemplo de Visibilidad de Widget")

        # Botón que inicia el proceso
        self.boton_iniciar = tk.Button(ventana, text="Iniciar Proceso", command=self.iniciar_proceso)
        self.boton_iniciar.grid(row=0, column=0, padx=10, pady=10)

        # Cuadro desplegable oculto inicialmente
        self.opciones_grabacion = ttk.Combobox(ventana, values=["Opción 1", "Opción 2", "Opción 3"], state="readonly")
        self.opciones_grabacion.current(0)  # Establecer la opción predeterminada

    def iniciar_proceso(self):
        # Simulación de un proceso (puedes remplazar esto con tu lógica de proceso)
        print("Proceso iniciado...")
        self.ventana.after(2000, self.mostrar_combobox)  # Mostrar combobox después de 2 segundos

    def mostrar_combobox(self):
        # Mostrar el cuadro desplegable
        print("Proceso alcanzado, mostrando opciones...")
        self.opciones_grabacion.grid(row=1, column=0, padx=10, pady=10)

# Crear y mostrar la ventana de la aplicación
ventana = tk.Tk()
app = MiApp(ventana)
ventana.mainloop()
