# TFM - Análisis de Hepatitis C con Machine Learning
## Guía de Uso Paso a Paso

---

## 📋 CONTENIDO DEL REPOSITORIO

Este código contiene el análisis completo del TFM incluyendo:
- **11 figuras** generadas con datos reales
- **5 modelos de Machine Learning** (Random Forest, Gradient Boosting, SVM, Logistic Regression, Naive Bayes)
- **Optimización de hiperparámetros** con Grid Search CV
- **Balanceo de clases** con SMOTE
- **Selección de características** con SFS

---

## 🚀 INSTRUCCIONES RÁPIDAS (Resumen)

```bash
# 1. Crear y activar entorno virtual
# Windows:
python -m venv env
env\Scripts\activate

# Linux/Mac:
python3 -m venv env
source env/bin/activate

# 2. Instalar dependencias desde requirements.txt
pip install -r requirements.txt

# 3. Descargar dataset hepatitisC.csv en la misma carpeta

# 4. Ejecutar el análisis
python codigo_tfm_analisis.py
```

---

## 📖 INSTRUCCIONES DETALLADAS PASO A PASO

### **PASO 1: Preparar el Entorno**

#### Opción A: Configuración con Virtualenv (Recomendado)

**En Windows:**

1. **Instalar Python** (versión 3.8 o superior):
   - Descargar desde: https://www.python.org/downloads/
   - **IMPORTANTE**: Marcar la casilla "Add Python to PATH" durante la instalación

2. **Abrir PowerShell o CMD**:
   - Windows: Presionar `Win + R`, escribir `powershell`, presionar Enter

3. **Navegar a la carpeta del proyecto**:
   ```powershell
   cd C:\Users\TuUsuario\Desktop\TFMDani
   ```

4. **Crear entorno virtual**:
   ```powershell
   python -m venv env
   ```

5. **Activar entorno virtual**:
   ```powershell
   env\Scripts\activate
   ```
   
   *Verás que el prompt cambia a algo como `(env) PS C:\...>`*

**En Linux/Mac:**

1. **Instalar Python** (si no está instalado):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3 python3-venv
   
   # macOS (con Homebrew)
   brew install python3
   ```

2. **Abrir Terminal**

3. **Navegar a la carpeta del proyecto**:
   ```bash
   cd ~/Desktop/TFMDani
   # O donde hayas descargado el proyecto
   ```

4. **Crear entorno virtual**:
   ```bash
   python3 -m venv env
   ```

5. **Activar entorno virtual**:
   ```bash
   source env/bin/activate
   ```
   
   *Verás que el prompt cambia a algo como `(env) $`*

#### Opción B: Usar Anaconda (Alternativa)

1. **Instalar Anaconda**: https://www.anaconda.com/download
2. **Abrir Anaconda Prompt (Windows) o Terminal (Mac/Linux)**
3. **Crear entorno virtual**:
   ```bash
   conda create -n tfm python=3.10
   conda activate tfm
   ```

---

### **PASO 2: Instalar Librerías Necesarias**

**IMPORTANTE**: Asegúrate de que el entorno virtual está activado (verás `(env)` en el prompt)

#### Usando requirements.txt (Recomendado)

En la terminal, ejecutar este comando:

```bash
pip install -r requirements.txt
```

Este comando instala automáticamente todas las dependencias necesarias con las versiones correctas.

```

**¿Qué instala cada librería?**
- `pandas`: Manejo de datos en tablas (DataFrames)
- `numpy`: Operaciones matemáticas con matrices
- `matplotlib`: Creación de gráficos y figuras
- `seaborn`: Visualizaciones estadísticas avanzadas
- `scikit-learn`: Algoritmos de Machine Learning
- `scipy`: Procesamiento de datos científicos
- `imbalanced-learn`: Técnicas de balanceo (SMOTE)

**Si hay errores de instalación**, probar:
```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

**Para desactivar el entorno virtual cuando termines**:
```bash
# Windows
deactivate

# Linux/Mac
deactivate
```

---

### **PASO 3: Obtener el Dataset**

1. **Descargar el archivo** `hepatitisC.csv` desde:
   - UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Hepatitis+C+Virus+(HCV)+for+Egyptian+patients
   - O usar el archivo proporcionado en el TFM

2. **Colocar el archivo** en la **misma carpeta** donde está el script (junto con `codigo_tfm_analisis.py`)

3. **Verificar que el archivo existe**:
   ```bash
   # Windows
   dir hepatitisC.csv
   
   # Linux/Mac
   ls hepatitisC.csv
   ```

---

### **PASO 4: Ejecutar el Código**

#### Método 1: Desde Terminal (Recomendado)

**En Windows (PowerShell o CMD)**:

```powershell
# 1. Navegar a la carpeta donde está el archivo
cd C:\Users\TuUsuario\Desktop\TFMDani

# 2. Activar el entorno virtual
env\Scripts\activate

# 3. Ejecutar el código
python codigo_tfm_analisis.py
```

**En Linux/Mac (Terminal)**:

```bash
# 1. Navegar a la carpeta donde está el archivo y activar el entorno virtual
source env/bin/activate

# 2. Ejecutar el código
python codigo_tfm_analisis.py
```

#### Método 2: Desde VS Code

1. Abrir VS Code
2. Archivo → Abrir Carpeta → Seleccionar carpeta con los archivos
3. Abrir `codigo_tfm_analisis.py`
4. **IMPORTANTE**: Seleccionar el intérprete de Python del virtualenv:
   - Ctrl+Shift+P (o Cmd+Shift+P en Mac)
   - Escribir "Python: Select Interpreter"
   - Elegir la opción que diga `./env/bin/python` o `env\Scripts\python.exe`
5. Presionar `F5` o el botón ▶️ "Run Python File"

#### Método 3: Desde Jupyter Notebook

```python
# En una celda de Jupyter:
%run codigo_tfm_analisis.py
```

---

### **PASO 5: Resultados Esperados**

Al ejecutar, verás en la terminal:

```
======================================================================
TFM - ANÁLISIS DE HEPATITIS C CON MACHINE LEARNING
======================================================================

======================================================================
1. CARGA Y PREPARACIÓN DE DATOS
======================================================================
Dataset cargado: 589 muestras, 14 variables
...
Preparación completada:
  Features: ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
  Muestras: 589
  Distribución clase: {0: 533, 1: 56}

✓ Figura 1 guardada: figura1_distribucion.png
✓ Figura 2 guardada: figura2_marcadores.png
...
```

---

### **PASO 6: Encontrar las Figuras Generadas**

Las figuras se guardan automáticamente en la **misma carpeta** donde ejecutaste el código:

| Archivo | Descripción |
|---------|-------------|
| `figura1_distribucion.png` | Distribución de pacientes por categoría |
| `figura2_marcadores.png` | Distribución de marcadores hepáticos |
| `figura3_correlacion.png` | Matriz de correlación de Pearson |
| `figura4_pca.png` | Análisis de Componentes Principales |
| `figura5_comparacion_base.png` | Comparación de métricas (Sin SMOTE) |
| `figura6_roc_base.png` | Curvas ROC (Sin SMOTE) |
| `figura7_importancia_base.png` | Importancia de variables (Gini) |
| `figura8_comparacion_smote.png` | Comparación de métricas (Con SMOTE) |
| `figura9_roc_smote.png` | Curvas ROC comparativas |
| `figura10_importancia_sfs.png` | Importancia con SFS (8 variables) |
| `figura11_comparacion_final.png` | Comparación final Sin vs Con SMOTE |

---

## ⚙️ CONFIGURACIÓN OPCIONAL

### **Cambiar el nombre del archivo CSV**

Si tu archivo CSV tiene otro nombre, editar la línea 529 del código:

```python
# Cambiar esto:
df = cargar_datos('hepatitisC.csv')

# Por esto (ejemplo):
df = cargar_datos('mis_datos.csv')
```

### **Cambiar la ruta del archivo**

```python
# Ruta absoluta (Windows)
df = cargar_datos(r'C:\Users\TuNombre\Documents\hepatitisC.csv')

# Ruta absoluta (Mac/Linux)
df = cargar_datos('/home/usuario/documentos/hepatitisC.csv')
```

### **Modificar parámetros de los modelos**

Editar las funciones de optimización (líneas 336-413):

```python
# Ejemplo: cambiar número de árboles en Random Forest
param_grid = {
    'n_estimators': [100, 200, 500],  # Aumentar valores
    'max_depth': [5, 10, 15, 20, None],  # Añadir más opciones
    # ... resto de parámetros
}
```

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### **Error: "ModuleNotFoundError: No module named 'pandas'" u otras librerías**

**Causa**: El entorno virtual no está activado, o las librerías no se instalaron correctamente.

**Solución**:
1. Verifica que el entorno virtual está activado (debe ver `(env)` en el prompt):
   ```bash
   # Windows
   env\Scripts\activate
   
   # Linux/Mac
   source env/bin/activate
   ```

2. Reinstala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### **Error: "El entorno virtual no se activa" (Windows)**

**Solución**:
Si obtienes un error como "cannot be loaded because running scripts is disabled on this system":

```powershell
# Ejecutar PowerShell como Administrador y escribir:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego intenta activar de nuevo:
```powershell
env\Scripts\activate
```

### **Error: "ModuleNotFoundError: No module named 'jupyter'"**

**Causa**: Jupyter no está instalado en el entorno virtual.

**Solución**:
```bash
pip install jupyter notebook
```

### **Error: "FileNotFoundError: hepatitisC.csv"**

**Causa**: El archivo CSV no está en la carpeta correcta.

**Solución**:
1. Verificar que `hepatitisC.csv` está en la misma carpeta que el código:
   ```bash
   # Windows
   dir hepatitisC.csv
   
   # Linux/Mac
   ls hepatitisC.csv
   ```

2. O especificar la ruta completa en el código (línea 529):
   ```python
   # Windows
   df = cargar_datos(r'C:\Ruta\Completa\hepatitisC.csv')
   
   # Linux/Mac
   df = cargar_datos('/home/usuario/ruta/hepatitisC.csv')
   ```

### **Las figuras no se generan / Error de permisos**

**Causa**: No hay permisos de escritura en la carpeta.

**Solución**:
1. Ejecutar terminal como Administrador (Windows)
2. O cambiar a una carpeta donde tengas permisos (ej: Escritorio)
   ```bash
   cd ~\Desktop  # Windows
   cd ~/Desktop  # Linux/Mac
   ```

### **Error: "MemoryError" o el programa se cierra**

**Causa**: El dataset es muy grande para la memoria RAM.

**Solución**:
El dataset de hepatitis C es pequeño (589 muestras), pero si usas otro dataset más grande:
- Cerrar otros programas
- Reducir `n_estimators` en los modelos
- Usar `n_jobs=1` en lugar de `n_jobs=-1`

### **Error: "requirements.txt not found"**

**Causa**: El archivo no existe en la carpeta actual.

**Solución**:
1. Asegúrate de estar en la carpeta correcta:
   ```bash
   # Verifica que estás en la carpeta del proyecto
   ls  # Linux/Mac
   dir  # Windows
   ```

2. El archivo `requirements.txt` debe estar visible

### **¿Cómo eliminar el entorno virtual si necesitas empezar de nuevo?**

```bash
# Windows
rmdir /s env

# Linux/Mac
rm -rf env
```

Luego crea uno nuevo:
```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

---

## 📊 EXPLICACIÓN DE LOS RESULTADOS

### **Métricas de Evaluación**

| Métrica | Qué mide | Valor Ideal |
|---------|----------|-------------|
| **Accuracy** | Porcentaje de aciertos totales | 1.0 (100%) |
| **Precision** | De los predichos positivos, cuántos son realmente positivos | 1.0 |
| **Recall** | De los realmente positivos, cuántos detectamos | 1.0 |
| **F1-Score** | Balance entre Precision y Recall | 1.0 |
| **AUC-ROC** | Capacidad de discriminación del modelo | 1.0 |
| **MCC** | Correlación de Matthews (balanceado) | 1.0 |

### **Qué significa SMOTE**

- **Problema**: El dataset está desbalanceado (533 sanos vs 56 enfermos)
- **Solución SMOTE**: Genera muestras sintéticas de la clase minoritaria para balancear
- **Resultado**: Mejora la detección de casos positivos (enfermos)

### **Qué significa SFS**

- **SFS** = Sequential Feature Selector
- **Función**: Selecciona las mejores variables automáticamente
- **En este TFM**: Selecciona 8 variables de las 11 disponibles

---

## 🎓 PARA INCLUIR EN LA MEMORIA DEL TFM

### **Citar el código en el documento**

Añadir en la sección de Metodología o Anexos:

> El código completo de este análisis está disponible en el siguiente repositorio:
> **[URL de tu repositorio GitHub/GitLab]**
> 
> El repositorio incluye:
> - Script Python completo (`codigo_tfm_analisis.py`)
> - Dataset utilizado (`hepatitisC.csv`)
> - Instrucciones de instalación y uso (`README.md`)
> - Todas las figuras generadas

### **Subir a GitHub (Opcional pero recomendado)**

1. Crear cuenta en https://github.com
2. Crear nuevo repositorio: `tfm-hepatitis-c`
3. Subir archivos:
   ```bash
   git init
   git add codigo_tfm_analisis.py hepatitisC.csv README.md
   git commit -m "Código TFM Hepatitis C"
   git push origin main
   ```

---

## 📞 CONTACTO Y SOPORTE

Si tienes problemas:
1. Verificar que todas las dependencias están instaladas
2. Comprobar que el archivo CSV tiene el formato correcto
3. Revisar los mensajes de error en la terminal

---

## 📝 RESUMEN DE COMANDOS

### **Windows (PowerShell o CMD)**

```powershell
# Crear entorno virtual
python -m venv env

# Activar entorno virtual
env\Scripts\activate

# Instalar dependencias desde requirements.txt
pip install -r requirements.txt

# Ejecutar el análisis
python codigo_tfm_analisis.py

# Ver figuras generadas
dir *.png

# Desactivar entorno virtual
deactivate
```

### **Linux/Mac (Terminal)**

```bash
# Crear entorno virtual
python3 -m venv env

# Activar entorno virtual
source env/bin/activate

# Instalar dependencias desde requirements.txt
pip install -r requirements.txt

# Ejecutar el análisis
python codigo_tfm_analisis.py

# Ver figuras generadas
ls *.png

# Desactivar entorno virtual
deactivate
```

### **Anaconda (Alternativa)**

```bash
# Crear entorno
conda create -n tfm python=3.10

# Activar entorno
conda activate tfm

# Instalar desde requirements.txt
pip install -r requirements.txt

# Ejecutar el análisis
python codigo_tfm_analisis.py

# Desactivar entorno
conda deactivate
```

---

**Versión del código**: 1.0  
**Fecha**: Abril 2025  
**Autor**: [Tu nombre]  
**Tutor**: [Nombre del tutor]
