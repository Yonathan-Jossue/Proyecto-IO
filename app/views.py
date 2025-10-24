import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from copy import deepcopy
from django import template
from copy import deepcopy
from fractions import Fraction
register = template.Library()

def home(request):
    return render(request, 'home.html')

def modulo1(request):
    return render(request, 'modulo1.html')

def modulo2(request):
    return render(request, 'modulo2.html')

def modulo3(request):
    return render(request, 'modulo3.html')

def modulo1_simplex(request):
    resultado = ''
    formulario_generado = False
    tablas_iteraciones = []

    # Variables por GET para generar formulario
    num_variables = request.GET.get('num_variables')
    num_restricciones = request.GET.get('num_restricciones')

    if num_variables and num_restricciones:
        num_variables = int(num_variables)
        num_restricciones = int(num_restricciones)
        formulario_generado = True
        variables = list(range(num_variables))
        restricciones = list(range(num_restricciones))
    else:
        num_variables = 2
        num_restricciones = 2
        variables = list(range(num_variables))
        restricciones = list(range(num_restricciones))

    if request.method == 'POST':
        # Obtener datos del POST
        num_variables = int(request.POST['num_variables'])
        num_restricciones = int(request.POST['num_restricciones'])
        variables = list(range(num_variables))
        restricciones = list(range(num_restricciones))

        # Función objetivo
        z = [float(request.POST[f'x{i}']) for i in variables]

        # Restricciones
        lista_restricciones = []
        for r in restricciones:
            coef = [float(request.POST[f'r{r}x{i}']) for i in variables]
            signo = request.POST[f'r{r}sign']
            rhs = float(request.POST[f'r{r}b'])
            lista_restricciones.append((coef, signo, rhs))

        # Construir tabla inicial simplex
        # Agregamos variables de holgura
        num_vars_totales = num_variables + num_restricciones
        tabla = []
        base = []
        for i, (coef, signo, rhs) in enumerate(lista_restricciones):
            fila = coef + [1 if j==i else 0 for j in range(num_restricciones)] + [rhs]
            tabla.append(fila)
            base.append(f"s{i+1}")

        # Fila Z
        z_fila = [-v for v in z] + [0]*num_restricciones + [0]
        tabla.append(z_fila)
        base.append("Z")

        encabezados = [f"x{i+1}" for i in range(num_variables)] + [f"s{i+1}" for i in range(num_restricciones)] + ["RHS"]

        # Función Simplex
        iteracion = 0
        max_iter = 20
        while True:
            # Guardar tabla actual
            tablas_iteraciones.append((base.copy(), [fila.copy() for fila in tabla], iteracion))

            # Última fila = Z
            z_row = tabla[-1][:-1]
            if all(c >= 0 for c in z_row):
                break  # óptimo alcanzado

            # Columna entrante (más negativa)
            col_entrante = z_row.index(min(z_row))

            # Razón mínima para determinar fila saliente
            ratios = []
            for i in range(len(tabla)-1):
                if tabla[i][col_entrante] > 0:
                    ratios.append(tabla[i][-1] / tabla[i][col_entrante])
                else:
                    ratios.append(float('inf'))
            if all(r == float('inf') for r in ratios):
                resultado += "<p>Problema no acotado</p>"
                break

            fila_saliente = ratios.index(min(ratios))

            # Pivot
            pivot = tabla[fila_saliente][col_entrante]
            tabla[fila_saliente] = [x / pivot for x in tabla[fila_saliente]]

            for i in range(len(tabla)):
                if i != fila_saliente:
                    factor = tabla[i][col_entrante]
                    tabla[i] = [tabla[i][j] - factor*tabla[fila_saliente][j] for j in range(len(tabla[0]))]

            # Actualizar variable básica
            base[fila_saliente] = encabezados[col_entrante]

            iteracion += 1
            if iteracion >= max_iter:
                resultado += "<p>Se alcanzó el límite de iteraciones.</p>"
                break

        # Generar HTML para mostrar tablas
        resultado = ''
        for base_fila, t, it in tablas_iteraciones:
            resultado += f'<h3>Iteración {it}</h3>'
            resultado += '<table class="simplex-table">'
            resultado += '<tr><th>Base</th>'
            for h in encabezados:
                resultado += f'<th>{h}</th>'
            resultado += '</tr>'
            for i, fila in enumerate(t):
                resultado += f'<tr><td>{base_fila[i]}</td>'
                for val in fila:
                    resultado += f'<td>{round(val,2)}</td>'
                resultado += '</tr>'
            resultado += '</table>'

        # Solución final
        sol = {f"x{i+1}":0 for i in range(num_variables)}
        for i, var in enumerate(base[:-1]):  # sin Z
            if var.startswith("x"):
                idx = int(var[1:])-1
                sol[var] = tabla[i][-1]
        resultado += '<h3>Solución óptima</h3>'
        resultado += '<ul>'
        for k,v in sol.items():
            resultado += f'<li>{k} = {round(v,2)}</li>'
        resultado += f'<li>Z = {round(tabla[-1][-1],2)}</li>'
        resultado += '</ul>'

        formulario_generado = True

    return render(request, 'simplex.html', {
        'formulario_generado': formulario_generado,
        'num_variables': num_variables,
        'num_restricciones': num_restricciones,
        'variables': variables,
        'restricciones': restricciones,
        'resultado': resultado
    })


def modulo_grafico(request):
    resultado = ''
    formulario_generado = False
    grafico_base64 = None

    # Solo trabajamos con 2 variables (x1, x2)
    num_restricciones = request.GET.get('num_restricciones')

    if num_restricciones:
        num_restricciones = int(num_restricciones)
        formulario_generado = True
        restricciones = list(range(num_restricciones))
    else:
        num_restricciones = 2
        restricciones = list(range(num_restricciones))

    if request.method == 'POST':
        num_restricciones = int(request.POST['num_restricciones'])
        restricciones = list(range(num_restricciones))

        # Función objetivo
        c1 = float(request.POST.get('c1', 1))
        c2 = float(request.POST.get('c2', 1))

        # Restricciones
        lista_restricciones = []
        for r in restricciones:
            a1 = float(request.POST[f'r{r}x1'])
            a2 = float(request.POST[f'r{r}x2'])
            b = float(request.POST[f'r{r}b'])
            lista_restricciones.append((a1, a2, b))

        # --- Generar gráfica ---
        x = np.linspace(0, 20, 400)
        plt.figure(figsize=(6, 6))

        # Dibujar restricciones
        for (a1, a2, b) in lista_restricciones:
            y = (b - a1*x) / a2
            plt.plot(x, y, label=f"{a1}x1 + {a2}x2 ≤ {b}")

        # Región factible aproximada
        y_limit = np.minimum.reduce([(b - a1*x)/a2 for (a1, a2, b) in lista_restricciones])
        y_limit = np.maximum(y_limit, 0)
        plt.fill_between(x, 0, y_limit, where=(y_limit >= 0), alpha=0.3)

        # --- Cálculo de soluciones óptimas ---
        puntos = [(0, 0)]
        # Intersecciones con ejes
        for (a1, a2, b) in lista_restricciones:
            if a1 != 0:  # intersección con x2=0
                puntos.append((b/a1, 0))
            if a2 != 0:  # intersección con x1=0
                puntos.append((0, b/a2))

        # Intersecciones entre rectas
        for i in range(len(lista_restricciones)):
            for j in range(i+1, len(lista_restricciones)):
                a1, a2, b1 = lista_restricciones[i]
                c1_, c2_, b2 = lista_restricciones[j]
                A = np.array([[a1, a2], [c1_, c2_]])
                B = np.array([b1, b2])
                try:
                    sol = np.linalg.solve(A, B)
                    puntos.append((sol[0], sol[1]))
                except np.linalg.LinAlgError:
                    pass  # rectas paralelas

        # Filtrar puntos factibles
        factibles = []
        for (x1, x2) in puntos:
            if x1 >= 0 and x2 >= 0:
                valido = True
                for (a1, a2, b) in lista_restricciones:
                    if a1*x1 + a2*x2 > b + 1e-6:
                        valido = False
                        break
                if valido:
                    factibles.append((x1, x2))

        # Evaluar función objetivo
        if factibles:
            valores = [(x1, x2, c1*x1 + c2*x2) for (x1, x2) in factibles]
            optimo = max(valores, key=lambda v: v[2])  # Maximización
            resultado = "<h3>Solución Óptima</h3>"
            resultado += f"<p>x1 = {round(optimo[0],2)}, x2 = {round(optimo[1],2)}</p>"
            resultado += f"<p>Z = {round(optimo[2],2)}</p>"
        else:
            resultado = "<p>No hay región factible.</p>"

        # Dibujar función objetivo en el gráfico
        z_line = (optimo[2] - c1*x) / c2 if factibles else (10 - c1*x) / c2
        plt.plot(x, z_line, 'r--', label="Función objetivo")

        # Estética
        plt.xlim(0, max(x))
        plt.ylim(0, 20)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Método gráfico - Programación lineal")
        plt.legend()

        # Guardar como base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        grafico_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        formulario_generado = True

    return render(request, 'grafico.html', {
        'formulario_generado': formulario_generado,
        'num_restricciones': num_restricciones,
        'restricciones': restricciones,
        'resultado': resultado,
        'grafico': grafico_base64
    })



M = 1e6  # Valor Gran M

def gran_m(request):
    resultado = ''
    formulario_generado = False
    tablas_iteraciones = []

    num_variables = request.GET.get('num_variables')
    num_restricciones = request.GET.get('num_restricciones')

    if num_variables and num_restricciones:
        num_variables = int(num_variables)
        num_restricciones = int(num_restricciones)
        formulario_generado = True
        variables = list(range(num_variables))
        restricciones = list(range(num_restricciones))
    else:
        num_variables = 2
        num_restricciones = 2
        variables = list(range(num_variables))
        restricciones = list(range(num_restricciones))

    if request.method == 'POST':
        try:
            num_variables = int(request.POST['num_variables'])
            num_restricciones = int(request.POST['num_restricciones'])
            variables = list(range(num_variables))
            restricciones = list(range(num_restricciones))

            # Función objetivo
            z = [float(request.POST[f'x{i}']) for i in variables]

            # Restricciones
            lista_restricciones = []
            for r in restricciones:
                coef = [float(request.POST[f'r{r}x{i}']) for i in variables]
                signo = request.POST[f'r{r}sign']
                rhs = float(request.POST[f'r{r}b'])
                lista_restricciones.append((coef, signo, rhs))

            # Construir tabla inicial con Gran M
            tabla, base, encabezados = construir_tabla_gran_m(lista_restricciones, z, num_variables)

            # Iteraciones Simplex
            iteracion = 0
            max_iter = 50
            while True:
                tablas_iteraciones.append((deepcopy(base), deepcopy(tabla), iteracion))

                # Última fila = Z
                z_row = tabla[-1][:-1]
                if all(c >= 0 for c in z_row):
                    break

                col_entrante = np.argmin(z_row)

                # Razón mínima
                ratios = []
                for i in range(len(tabla)-1):
                    ratios.append(tabla[i][-1]/tabla[i][col_entrante] if tabla[i][col_entrante]>0 else float('inf'))

                if all(r == float('inf') for r in ratios):
                    resultado += "<p>Problema no acotado</p>"
                    break

                fila_saliente = ratios.index(min(ratios))

                pivot = tabla[fila_saliente][col_entrante]
                tabla[fila_saliente] = [x/pivot for x in tabla[fila_saliente]]

                for i in range(len(tabla)):
                    if i != fila_saliente:
                        factor = tabla[i][col_entrante]
                        tabla[i] = [tabla[i][j]-factor*tabla[fila_saliente][j] for j in range(len(tabla[0]))]

                base[fila_saliente] = encabezados[col_entrante]
                iteracion += 1
                if iteracion >= max_iter:
                    resultado += "<p>Límite de iteraciones alcanzado</p>"
                    break

            # Generar HTML
            resultado = generar_tablas_html(tablas_iteraciones, encabezados, num_variables)

        except Exception as e:
            resultado = f"<p>Error: {str(e)}</p>"

        formulario_generado = True

    return render(request, 'gran_m.html', {
        'formulario_generado': formulario_generado,
        'num_variables': num_variables,
        'num_restricciones': num_restricciones,
        'variables': variables,
        'restricciones': restricciones,
        'resultado': resultado
    })


# =========================
# Funciones auxiliares
# =========================



def construir_tabla_gran_m(lista_restricciones, z, num_vars):
    M = 1e6
    # Contar tipos de restricciones
    num_slack = sum(1 for _, s, _ in lista_restricciones if s == "<=")
    num_excess = sum(1 for _, s, _ in lista_restricciones if s == ">=")
    num_equal = sum(1 for _, s, _ in lista_restricciones if s == "=")
    num_artificial = num_excess + num_equal

    total_columnas = num_vars + num_slack + num_excess + num_artificial + 1
    tabla = []
    base = []
    slack_idx = 0
    excess_idx = 0
    art_idx = 0

    for coef, signo, rhs in lista_restricciones:
        fila = [0]*total_columnas
        for i in range(num_vars):
            fila[i] = coef[i]

        col_idx = num_vars
        if signo == "<=":
            fila[col_idx + slack_idx] = 1
            base.append(f"s{slack_idx+1}")
            slack_idx += 1
        elif signo == ">=":
            fila[col_idx + slack_idx + excess_idx] = -1
            fila[col_idx + num_slack + num_excess + art_idx] = 1
            base.append(f"a{art_idx+1}")
            excess_idx += 1
            art_idx += 1
        elif signo == "=":
            fila[col_idx + num_slack + num_excess + art_idx] = 1
            base.append(f"a{art_idx+1}")
            art_idx += 1

        fila[-1] = rhs
        tabla.append(fila)

    # Encabezados
    encabezados = [f"x{i+1}" for i in range(num_vars)]
    for i in range(num_slack):
        encabezados.append(f"s{i+1}")
    for i in range(num_excess):
        encabezados.append(f"e{i+1}")
    for i in range(num_artificial):
        encabezados.append(f"a{i+1}")
    encabezados.append("RHS")

    # Fila Z con Gran M
    fila_z = [-v for v in z] + [0]*(total_columnas - num_vars - 1) + [0]

    # Ajustar Z sumando -M*artificial si está en la base
    for i, var in enumerate(base):
        if var.startswith('a'):
            fila_z = [fila_z[j] - M*tabla[i][j] for j in range(len(fila_z))]

    tabla.append(fila_z)
    base.append("Z")

    return tabla, base, encabezados

def generar_tablas_html(tablas_iteraciones, encabezados, num_variables):
    """
    Genera el HTML de las tablas por iteración y la solución óptima.
    """
    resultado = ''
    for base_fila, t, it in tablas_iteraciones:
        resultado += f'<h3>Iteración {it}</h3>'
        resultado += '<table class="simplex-table">'
        resultado += '<tr><th>Base</th>'
        for h in encabezados:
            resultado += f'<th>{h}</th>'
        resultado += '</tr>'
        for i, fila in enumerate(t):
            resultado += f'<tr><td>{base_fila[i]}</td>'
            for val in fila:
                resultado += f'<td>{round(val,5)}</td>'
            resultado += '</tr>'
        resultado += '</table>'

    # Solución final
    sol = {f"x{i+1}": 0 for i in range(num_variables)}
    final_tabla = tablas_iteraciones[-1][1]
    final_base = tablas_iteraciones[-1][0]

    for i, var in enumerate(final_base[:-1]):
        if var.startswith("x"):
            sol[var] = final_tabla[i][-1]

    # Z final
    z_final = final_tabla[-1][-1]
    
    resultado += '<h3>Solución óptima</h3>'
    resultado += '<ul>'
    for k,v in sol.items():
        resultado += f'<li>{k} = {round(v,5)}</li>'
    resultado += f'<li>Z = {round(z_final,5)}</li>'
    resultado += '</ul>'

    return resultado


def dos_fases(request):
    resultado = ''
    formulario_generado = False
    tablas_iteraciones = []

    # Obtener GET para generar formulario
    num_variables = request.GET.get('num_variables')
    num_restricciones = request.GET.get('num_restricciones')

    if num_variables and num_restricciones:
        num_variables = int(num_variables)
        num_restricciones = int(num_restricciones)
        formulario_generado = True
        variables = range(num_variables)
        restricciones = range(num_restricciones)
    else:
        num_variables = 2
        num_restricciones = 2
        variables = range(num_variables)
        restricciones = range(num_restricciones)

    if request.method == 'POST':
        try:
            num_variables = int(request.POST['num_variables'])
            num_restricciones = int(request.POST['num_restricciones'])
            variables = range(num_variables)
            restricciones = range(num_restricciones)

            # Tipo de problema: max o min
            tipo = request.POST.get('tipo', 'max')  # por defecto max

            # Función objetivo
            z = [float(request.POST[f'x{i}']) for i in variables]

            # Restricciones
            lista_restricciones = []
            for r in restricciones:
                coef = [float(request.POST[f'r{r}x{i}']) for i in variables]
                signo = request.POST[f'r{r}sign']
                rhs = float(request.POST[f'r{r}b'])
                lista_restricciones.append((coef, signo, rhs))

            # Construcción tabla inicial Fase 1
            tabla = []
            base = []
            encabezados = [f"x{i+1}" for i in range(num_variables)]
            artificiales = []
            slack_count = 0
            excess_count = 0

            for i, (coef, signo, rhs) in enumerate(lista_restricciones):
                fila = coef.copy()
                fila += [0]*slack_count
                fila += [0]*excess_count

                if signo == "<=":
                    fila.append(1)
                    base.append(f"S{slack_count+1}")
                    encabezados.append(f"S{slack_count+1}")
                    slack_count += 1
                elif signo == ">=":
                    fila.append(-1)
                    encabezados.append(f"E{excess_count+1}")
                    excess_count += 1
                    fila.append(1)
                    base.append(f"R{i+1}")
                    encabezados.append(f"R{i+1}")
                    artificiales.append(f"R{i+1}")
                elif signo == "=":
                    fila.append(1)
                    base.append(f"R{i+1}")
                    encabezados.append(f"R{i+1}")
                    artificiales.append(f"R{i+1}")

                fila.append(rhs)
                tabla.append(fila)

            # Igualar columnas
            max_cols = max(len(f) for f in tabla)
            for f in tabla:
                while len(f) < max_cols:
                    f.insert(-1, 0)

            # Fila W Fase 1
            w_fila = [0]*max_cols
            for idx, var in enumerate(base):
                if var in artificiales:
                    w_fila = [w_fila[j] - tabla[idx][j] for j in range(len(w_fila))]
            tabla.append(w_fila)
            base.append("W")

            # Iteraciones Fase 1
            iteracion = 0
            max_iter = 50
            while True:
                tablas_iteraciones.append((base.copy(), [f.copy() for f in tabla], f"F1-{iteracion}"))
                w_row = tabla[-1][:-1]
                if all(c >= 0 for c in w_row):
                    break
                col_entrante = w_row.index(min(w_row))
                ratios = []
                for i in range(len(tabla)-1):
                    ratios.append(tabla[i][-1]/tabla[i][col_entrante] if tabla[i][col_entrante]>0 else float('inf'))
                if all(r==float('inf') for r in ratios):
                    resultado += "<p class='error'>⚠️ Problema no acotado (Fase 1)</p>"
                    break
                fila_saliente = ratios.index(min(ratios))
                pivot = tabla[fila_saliente][col_entrante]
                tabla[fila_saliente] = [x/pivot for x in tabla[fila_saliente]]
                for i in range(len(tabla)):
                    if i != fila_saliente:
                        factor = tabla[i][col_entrante]
                        tabla[i] = [tabla[i][j]-factor*tabla[fila_saliente][j] for j in range(len(tabla[0]))]
                base[fila_saliente] = encabezados[col_entrante]
                iteracion += 1
                if iteracion >= max_iter:
                    resultado += "<p class='error'>⚠️ Límite de iteraciones Fase 1</p>"
                    break

            # Fase 2: eliminar fila W y columnas artificiales
            tabla.pop()
            base.pop()
            art_indices = [encabezados.index(a) for a in artificiales if a in encabezados]
            for f in tabla:
                for idx in sorted(art_indices, reverse=True):
                    if idx < len(f)-1:  # <-- mantener RHS
                        f.pop(idx)
            for a in artificiales:
                if a in encabezados:
                    encabezados.remove(a)

            # Construir fila Z correctamente (con RHS)
            z_fila = [0]*(len(encabezados)+1)
            for i in range(num_variables):
                if f"x{i+1}" in encabezados:
                    idx = encabezados.index(f"x{i+1}")
                    z_fila[idx] = -z[i] if tipo=='max' else z[i]
            tabla.append(z_fila)
            base.append("Z")

            # Iteraciones Fase 2
            iteracion = 0
            while True:
                tablas_iteraciones.append((base.copy(), [f.copy() for f in tabla], f"F2-{iteracion}"))
                z_row = tabla[-1][:-1]
                if all(c >= 0 for c in z_row):
                    break
                col_entrante = z_row.index(min(z_row))
                ratios = []
                for i in range(len(tabla)-1):
                    ratios.append(tabla[i][-1]/tabla[i][col_entrante] if tabla[i][col_entrante]>0 else float('inf'))
                if all(r==float('inf') for r in ratios):
                    resultado += "<p class='error'>⚠️ Problema no acotado (Fase 2)</p>"
                    break
                fila_saliente = ratios.index(min(ratios))
                pivot = tabla[fila_saliente][col_entrante]
                tabla[fila_saliente] = [x/pivot for x in tabla[fila_saliente]]
                for i in range(len(tabla)):
                    if i != fila_saliente:
                        factor = tabla[i][col_entrante]
                        tabla[i] = [tabla[i][j]-factor*tabla[fila_saliente][j] for j in range(len(tabla[0]))]
                base[fila_saliente] = encabezados[col_entrante]
                iteracion += 1
                if iteracion >= max_iter:
                    resultado += "<p class='error'>⚠️ Límite de iteraciones Fase 2</p>"
                    break

            # Mostrar tablas y solución
            resultado = ''
            for base_fila, t, it in tablas_iteraciones:
                resultado += f'<h3>🧮 Iteración {it}</h3>'
                resultado += '<table class="simplex-table">'
                resultado += '<tr><th>Base</th>'
                for h in encabezados:
                    resultado += f'<th>{h}</th>'
                resultado += '<th>RHS</th></tr>'
                for i, fila in enumerate(t):
                    resultado += f'<tr><td>{base_fila[i]}</td>'
                    for val in fila:
                        resultado += f'<td>{round(val,5)}</td>'
                    resultado += '</tr>'
                resultado += '</table>'

            # Solución final
            sol = {f"x{i+1}":0 for i in range(num_variables)}
            for i, var in enumerate(base[:-1]):
                if var.startswith("x"):
                    sol[var] = tabla[i][-1]

            resultado += '<h3>✅ Solución óptima</h3><ul>'
            for k,v in sol.items():
                resultado += f'<li>{k} = {round(v,5)}</li>'
            resultado += f'<li>Z = {round(tabla[-1][-1],5)}</li>'
            resultado += '</ul>'

        except Exception as e:
            resultado = f"<p class='error'>❌ Error: {e}</p>"
            formulario_generado = True

    return render(request, 'dos_fases.html',{
        'resultado': resultado,
        'formulario_generado': formulario_generado,
        'num_variables': num_variables,
        'num_restricciones': num_restricciones,
        'variables': variables,
        'restricciones': restricciones
    })


def modulo_dual(request):
    """
    Recibe el problema primal, lo convierte correctamente a dual
    y decide el método adecuado para resolverlo.
    """
    resultado = ''
    formulario_generado = False

    num_variables = request.GET.get('num_variables')
    num_restricciones = request.GET.get('num_restricciones')

    if num_variables and num_restricciones:
        num_variables = int(num_variables)
        num_restricciones = int(num_restricciones)
        formulario_generado = True
        variables = range(num_variables)
        restricciones = range(num_restricciones)
    else:
        num_variables = 2
        num_restricciones = 2
        variables = range(num_variables)
        restricciones = range(num_restricciones)

    if request.method == 'POST':
        try:
            tipo = request.POST.get('tipo', 'max')  # max o min
            num_variables = int(request.POST['num_variables'])
            num_restricciones = int(request.POST['num_restricciones'])
            variables = range(num_variables)
            restricciones = range(num_restricciones)

            # Función objetivo
            c = [float(request.POST[f'x{i}']) for i in variables]

            # Restricciones
            A = []
            b = []
            signos = []
            for r in restricciones:
                coef = [float(request.POST[f'r{r}x{i}']) for i in variables]
                A.append(coef)
                signos.append(request.POST[f'r{r}sign'])
                b.append(float(request.POST[f'r{r}b']))

            # ======================
            # Conversión a problema dual (forma algebraica)
            # ======================

            tipo_dual = 'min' if tipo == 'max' else 'max'

            # Transponer matriz de coeficientes
            A_T = list(map(list, zip(*A)))

            # Función objetivo dual (coeficientes de b)
            dual_obj = b

            # Restricciones del dual
            restricciones_dual = []
            for i in range(num_variables):
                expr = [f"{A_T[i][j]}y{j+1}" for j in range(num_restricciones)]
                # Determinar signo según tipo del primal
                if tipo == 'max':
                    if all(s == "=" for s in signos):
                        signo_dual = "="
                    else:
                        signo_dual = ">=" if any(s == "<=" for s in signos) else "<="
                else:
                    signo_dual = "<=" if any(s == "<=" for s in signos) else ">="
                restricciones_dual.append(f"{' + '.join(expr)} {signo_dual} {c[i]}")

            # Determinar método adecuado
            if any("=" in r for r in restricciones_dual) or any(">=" in r for r in restricciones_dual):
                metodo = "dos_fases" if any("=" in r for r in restricciones_dual) else "gran_m"
            else:
                metodo = "simplex"

            # ======================
            # Mostrar el dual generado de forma más bonita
            # ======================
            resultado += "<div style='border:2px solid #ccc; padding:15px; border-radius:10px; background-color:#f9f9f9;'>"
            resultado += f"<h3 style='color:#333;'>Problema Dual Generado</h3>"

            # Función objetivo dual
            resultado += "<p><b>Función objetivo:</b></p>"
            resultado += "<p style='margin-left: 20px;'>"
            resultado += f"{tipo_dual.upper()} W = "
            resultado += " + ".join([f"<span style='color:blue'>{dual_obj[i]}y<sub>{i+1}</sub></span>" for i in range(len(dual_obj))])
            resultado += "</p>"

            # Restricciones duales
            resultado += "<p><b>Sujeto a:</b></p>"
            resultado += "<ul>"
            for restr in restricciones_dual:
                restr_html = restr
                for i in range(num_restricciones):
                    restr_html = restr_html.replace(f"y{i+1}", f"y<sub>{i+1}</sub>")
                resultado += f"<li>{restr_html}</li>"
            resultado += "</ul>"

            # Variables (y) y su signo según restricción primal
            resultado += "<p><b>Variables:</b> "
            vars_html = []
            for j in range(num_restricciones):
                if signos[j] == "=":
                    vars_html.append(f"y<sub>{j+1}</sub> libre")
                elif signos[j] == "<=":
                    vars_html.append(f"y<sub>{j+1}</sub> ≥ 0")
                else:  # ">="
                    vars_html.append(f"y<sub>{j+1}</sub> ≤ 0")
            resultado += ", ".join(vars_html)
            resultado += "</p>"

            # Método sugerido
            resultado += f"<p><b>Método sugerido:</b> <span style='color:green'>{metodo.upper()}</span></p>"
            resultado += "</div>"

        except Exception as e:
            resultado = f"<p style='color:red;'>Error: {str(e)}</p>"
        formulario_generado = True

    return render(request, 'dual.html', {
        'resultado': resultado,
        'formulario_generado': formulario_generado,
        'num_variables': num_variables,
        'num_restricciones': num_restricciones,
        'variables': range(num_variables),
        'restricciones': range(num_restricciones)
    })

def modulo_inventario(request):
    """
    Módulo de Inventarios (Modelo EOQ y Punto de Reorden)
    Muestra el cálculo con un diseño más atractivo y organizado.
    """
    resultado = ""
    formulario_generado = False

    if request.method == "POST":
        try:
            # Obtener datos
            D = float(request.POST.get("demanda_anual"))
            Cp = float(request.POST.get("costo_pedido"))
            Ch = float(request.POST.get("costo_mantener"))
            d = float(request.POST.get("demanda_diaria"))
            L = float(request.POST.get("lead_time"))
            SS = float(request.POST.get("seguridad"))

            from math import sqrt
            Q = sqrt((2 * D * Cp) / Ch)
            R = d * L + SS

            # Resultado con estilo
            resultado = f"""
            <div style='
                background-color:#ffffff;
                border-radius:15px;
                box-shadow:0 0 15px rgba(0,0,0,0.1);
                padding:25px;
                margin-top:20px;
                line-height:1.6;
            '>
                <h3 style='color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:8px;'>📘 Procedimiento del Modelo EOQ y Punto de Reorden</h3>

                <section style='margin-top:15px;'>
                    <h4 style='color:#2980b9;'>1️⃣ Cálculo de la Cantidad Económica de Pedido (EOQ)</h4>
                    <p><b>Fórmula:</b> Q = √((2 × D × Cp) / Ch)</p>
                    <p><b>Sustituyendo:</b> Q = √((2 × {D:.2f} × {Cp:.2f}) / {Ch:.2f})</p>
                    <div style='background:#eaf2f8; padding:10px; border-radius:8px;'>
                        <b>Resultado:</b> Q = <span style='color:#16a085; font-size:1.2em;'>{Q:.2f}</span> unidades
                    </div>
                </section>

                <section style='margin-top:20px;'>
                    <h4 style='color:#2980b9;'>2️⃣ Cálculo del Punto de Reorden (R)</h4>
                    <p><b>Fórmula:</b> R = (d × L) + SS</p>
                    <p><b>Sustituyendo:</b> R = ({d:.2f} × {L:.2f}) + {SS:.2f}</p>
                    <div style='background:#eaf2f8; padding:10px; border-radius:8px;'>
                        <b>Resultado:</b> R = <span style='color:#c0392b; font-size:1.2em;'>{R:.2f}</span> unidades
                    </div>
                </section>

                <section style='margin-top:25px; background:#f9f9f9; padding:15px; border-radius:10px; border-left:4px solid #2ecc71;'>
                    <h4 style='color:#27ae60;'>📊 Interpretación</h4>
                    <p>✔ Se recomienda realizar un pedido de <b>{Q:.2f}</b> unidades cada vez.</p>
                    <p>✔ El pedido debe realizarse cuando el inventario llegue a <b>{R:.2f}</b> unidades.</p>
                </section>
            </div>
            """

            formulario_generado = True

        except Exception as e:
            resultado = f"<p style='color:red;'>Error: {str(e)}</p>"

    return render(request, "inventario.html", {
        "resultado": resultado,
        "formulario_generado": formulario_generado
    })
