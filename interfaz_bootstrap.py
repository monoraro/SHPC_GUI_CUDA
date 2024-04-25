import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

class MyApp:
    def __init__(self, ventana):
        self.ventana = ventana
        ventana.title("Captura y Transformación de Cámara")
        ventana.geometry("1920x1080")
        # Establecer el tema "cosmo" al inicio
        ventana.style = ttkb.Style(theme="cosmo")
        # Crear frame principal
        self.main_frame = ttk.Frame(ventana, borderwidth=1, relief="solid")
        self.main_frame.pack(expand=True, fill="both")

        # Configurar las columnas para que se expandan por toda la longitud de la ventana
        for i in range(10):
            ventana.grid_columnconfigure(i, weight=1)

        # Título en la parte superior
        self.titulo_label = ttk.Label(self.main_frame, text="Aplicacion de reconstrucción", font=("Helvetica", 20, "bold"))
        self.titulo_label.grid(row=0, column=0, columnspan=10, pady=(10, 20), sticky="ew")
        self.hline = ttk.Separator(self.main_frame, orient="horizontal")
        self.hline.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(10, 10))

        # Selector de estilos en la parte superior
        ttk.Label(self.main_frame, text="Seleccionar estilo:").grid(row=1, column=0, padx=10, sticky="e")
        self.style_selector = ttk.Combobox(self.main_frame, values=["Estilo 1", "Estilo 2", "Estilo 3"], state="readonly")
        self.style_selector.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Cuadros de texto en el centro
        self.cuadro_texto1 = ttk.Entry(self.main_frame)
        self.cuadro_texto1.grid(row=5, column=3, padx=10, pady=10, sticky="ew")
        
        self.cuadro_texto2 = ttk.Entry(self.main_frame)
        self.cuadro_texto2.grid(row=5, column=6, padx=10, pady=10, sticky="ew")

        self.cuadro_texto3 = ttk.Entry(self.main_frame)
        self.cuadro_texto3.grid(row=8, column=3, padx=10, pady=10, sticky="ew")

        self.cuadro_texto4 = ttk.Entry(self.main_frame)
        self.cuadro_texto4.grid(row=8, column=6, padx=10, pady=10, sticky="ew")

# Creación de la ventana y de la aplicación
root = tk.Tk()
app = MyApp(root)
root.mainloop()
