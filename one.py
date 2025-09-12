#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot_direction_field(
    f,
    xmin,
    xmax,
    ymin,
    ymax,
    xstep,
    ystep,
    field_type="F",
    streamlines=False,
    arrow_scale=1.0,
    arrow_width=0.003,
    title="Campo de Direcciones",
    figsize=(10, 8),
):
    """
    Graficaelcampo de direcciones para una ecuación diferencial de primer orden.

    Parámetros:
    -----------
    f : function
        Función que define la ecuación diferencial dy/dx = f(x,y)
        o función auxiliar F(x,y) = (expr1, expr2) para campo vectorial
    xmin, xmax : float
        Límites del eje x
    ymin, ymax : float
        Límites del eje y
    xstep, ystep : float
        Separación de puntos en los ejes x e y
    field_type : str
        'F' para campo original, 'N' para campo unitario normalizado
    streamlines : bool
        Si True, grafica las líneas de flujo
    arrow_scale : float
        Factor de escala para las flechas
    arrow_width : float
        Grosor de las flechas
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figura con el campo de direcciones
    """

    # Crear la rejilla de puntos
    x = np.linspace(xmin, xmax, int((xmax - xmin) / xstep) + 1)
    y = np.linspace(ymin, ymax, int((ymax - ymin) / ystep) + 1)
    X, Y = np.meshgrid(x, y)

    # Evaluar el campo vectorial
    try:
        # Intentar evaluar f como campo vectorial F(x,y) = (expr1, expr2)
        field_result = f(X, Y)
        if isinstance(field_result, tuple) and len(field_result) == 2:
            U, V = field_result
        else:
            # Si no es tupla, asumir que es dy/dx = f(x,y)
            U = np.ones_like(X)  # dx/dt = 1
            V = field_result  # dy/dt = f(x,y)
    except:
        # Fallback: evaluar punto por punto
        U = np.ones_like(X)
        V = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    result = f(X[i, j], Y[i, j])
                    if isinstance(result, tuple):
                        U[i, j], V[i, j] = result
                    else:
                        U[i, j] = 1
                        V[i, j] = result
                except:
                    U[i, j] = 0
                    V[i, j] = 0

    # Aplicar normalización si se requiere campo unitario
    if field_type.upper() == "N":
        # Calcular magnitud
        magnitude = np.sqrt(U**2 + V**2)
        # Evitar división por cero
        magnitude = np.where(magnitude == 0, 1, magnitude)
        # Normalizar
        U = U / magnitude
        V = V / magnitude

    # Crear la figura
    fig, ax = plt.subplots(figsize=figsize)

    # Graficar el campo de direcciones
    quiver_plot = ax.quiver(
        X,
        Y,
        U,
        V,
        scale=1 / arrow_scale,
        width=arrow_width,
        angles="xy",
        scale_units="xy",
        alpha=0.7,
        color="blue",
    )

    # Graficar líneas de flujo si se solicita
    if streamlines:
        try:
            # Crear una rejilla más densa para las streamlines
            x_stream = np.linspace(xmin, xmax, 50)
            y_stream = np.linspace(ymin, ymax, 50)
            X_stream, Y_stream = np.meshgrid(x_stream, y_stream)

            # Evaluar el campo en la rejilla densa
            try:
                stream_result = f(X_stream, Y_stream)
                if isinstance(stream_result, tuple):
                    U_stream, V_stream = stream_result
                else:
                    U_stream = np.ones_like(X_stream)
                    V_stream = stream_result
            except:
                U_stream = np.ones_like(X_stream)
                V_stream = np.zeros_like(Y_stream)
                for i in range(X_stream.shape[0]):
                    for j in range(X_stream.shape[1]):
                        try:
                            result = f(X_stream[i, j], Y_stream[i, j])
                            if isinstance(result, tuple):
                                U_stream[i, j], V_stream[i, j] = result
                            else:
                                U_stream[i, j] = 1
                                V_stream[i, j] = result
                        except:
                            U_stream[i, j] = 0
                            V_stream[i, j] = 0

            ax.streamplot(
                X_stream,
                Y_stream,
                U_stream,
                V_stream,
                color="red",
                alpha=0.6,
                linewidth=1,
                density=1.5,
            )
        except Exception as e:
            print(f"No se pudieron graficar las líneas de flujo: {e}")

    # Configurar el gráfico
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"{title}\n(Campo {'Unitario' if field_type.upper() == 'N' else 'Original'})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Añadir información adicional
    info_text = f"Rejilla: {len(x)}×{len(y)} puntos\nTipo: Campo {'Normalizado' if field_type.upper() == 'N' else 'Original'}"
    if streamlines:
        info_text += "\nLíneas de flujo: Sí"

    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=bbox_props,
    )

    plt.tight_layout()
    return fig


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================


# Ejemplo 1: Ecuación diferencial dy/dx = -x/y (familia de círculos)
def ejemplo1(x, y):
    """Ecuación: dy/dx = -x/y"""
    # Evitar división por cero
    y_safe = np.where(np.abs(y) < 1e-10, 1e-10, y)
    return -x / y_safe


# Ejemplo 2: Campo vectorial espiral F(x,y) = (-y, x)
def ejemplo2(x, y):
    """Campo vectorial: F(x,y) = (-y, x)"""
    return (-y, x)


# Ejemplo 3: Ecuación logística dy/dx = y(1-y)
def ejemplo3(x, y):
    """Ecuación logística: dy/dx = y(1-y)"""
    return y * (1 - y)


# Ejemplo 4: Campo de Van der Pol
def ejemplo4(x, y):
    """Oscilador de Van der Pol simplificado"""
    mu = 1.0
    return (y, mu * (1 - x**2) * y - x)


# ============================================================================
# DEMOSTRACIONES
# ============================================================================

if __name__ == "__main__":

    print("=== DEMOSTRACIÓN: CAMPOS DE DIRECCIONES ===\n")

    # Demostración 1: Ecuación dy/dx = -x/y
    print("1. Ecuación diferencial: dy/dx = -x/y")
    print("   (Familia de circunferencias)")

    fig1 = plot_direction_field(
        f=ejemplo1,
        xmin=-3,
        xmax=3,
        ymin=-3,
        ymax=3,
        xstep=0.3,
        ystep=0.3,
        field_type="F",
        streamlines=True,
        arrow_scale=0.8,
        title="Ejemplo 1: dy/dx = -x/y",
    )
    plt.show()

    # Demostración 2: Campo vectorial espiral
    print("\n2. Campo vectorial: F(x,y) = (-y, x)")
    print("   (Campo rotacional)")

    fig2 = plot_direction_field(
        f=ejemplo2,
        xmin=-2,
        xmax=2,
        ymin=-2,
        ymax=2,
        xstep=0.2,
        ystep=0.2,
        field_type="N",  # Campo normalizado
        streamlines=True,
        arrow_scale=1.0,
        title="Ejemplo 2: F(x,y) = (-y, x)",
    )
    plt.show()

    # Demostración 3: Comparación de campos original vs normalizado
    print("\n3. Comparación: Campo original vs campo normalizado")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Campo original
    fig3a = plot_direction_field(
        f=ejemplo4,
        xmin=-2,
        xmax=2,
        ymin=-2,
        ymax=2,
        xstep=0.25,
        ystep=0.25,
        field_type="F",
        streamlines=False,
        title="Van der Pol - Campo Original",
    )

    # Campo normalizado
    fig3b = plot_direction_field(
        f=ejemplo4,
        xmin=-2,
        xmax=2,
        ymin=-2,
        ymax=2,
        xstep=0.25,
        ystep=0.25,
        field_type="N",
        streamlines=True,
        title="Van der Pol - Campo Normalizado",
    )

    plt.show()

    print("\n=== ANÁLISIS DE LOS EJEMPLOS ===")
    print("• Ejemplo 1: Muestra órbitas circulares características de dy/dx = -x/y")
    print("• Ejemplo 2: Campo rotacional puro con líneas de flujo circulares")
    print("• Ejemplo 3: Comparación entre campo original (magnitudes variables)")
    print("             y campo normalizado (direcciones solamente)")
    print("\nLa función permite gran flexibilidad en la visualización de")
    print("ecuaciones diferenciales y campos vectoriales.")

# ============================================================================
# FUNCIONES AUXILIARES ADICIONALES
# ============================================================================


def crear_campo_personalizado(expresion_x, expresion_y):
    """
    Crea una función de campo vectorial a partir de expresiones simbólicas.

    Ejemplo de uso:
    campo = crear_campo_personalizado("y", "-x + y**2")
    """

    def campo(x, y):
        # Reemplazar variables en las expresiones
        expr_x_eval = eval(expresion_x.replace("x", "x").replace("y", "y"))
        expr_y_eval = eval(expresion_y.replace("x", "x").replace("y", "y"))
        return (expr_x_eval, expr_y_eval)

    return campo


def analizar_puntos_equilibrio(f, xmin, xmax, ymin, ymax, tolerancia=1e-6):
    """
    Encuentra aproximadamente los puntos de equilibrio de un campo vectorial.
    """
    puntos_equilibrio = []

    # Rejilla para búsqueda
    x_test = np.linspace(xmin, xmax, 50)
    y_test = np.linspace(ymin, ymax, 50)

    for xi in x_test:
        for yi in y_test:
            try:
                result = f(xi, yi)
                if isinstance(result, tuple):
                    u, v = result
                    if abs(u) < tolerancia and abs(v) < tolerancia:
                        puntos_equilibrio.append((xi, yi))
                else:
                    if abs(result) < tolerancia:
                        puntos_equilibrio.append((xi, yi))
            except:
                continue

    return puntos_equilibrio
