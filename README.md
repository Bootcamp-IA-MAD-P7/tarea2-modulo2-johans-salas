# 📊 Tarea: Investigación y Desarrollo de un Modelo de Regresión Lineal Simple

Los coders deben investigar cómo se construye y evalúa un modelo de **Regresión Lineal Simple** aplicando Python y librerías comunes (pandas, scikit-learn, matplotlib, seaborn, 
etc.). 


## Parte 1: Fundamentos de Regresión

Cada estudiante debe investigar y responder las siguientes preguntas: 

**1. ¿Qué es la regresión y cuál es su propósito en el Machine Learning?**

La **regresión** es una técnica de aprendizaje supervisado cuyo objetivo es predecir un **valor numérico continuo** a partir de una o más variables de entrada (features). El modelo aprende la relación matemática entre las variables independientes (X) y la variable dependiente (y) a partir de datos históricos.

**Ejemplos de uso:**
- Predecir el precio de una vivienda en función de su tamaño y ubicación.
- Estimar las ventas futuras de un producto.
- Pronosticar la temperatura de mañana.

**Diferencias entre regresión y clasificación:**

| Característica | Regresión | Clasificación |
|---|---|---|
| **Salida** | Valor numérico continuo | Categoría / etiqueta discreta |
| **Ejemplo de pregunta** | ¿Cuánto costará esta casa? | ¿Es spam o no es spam? |
| **Métricas típicas** | MSE, RMSE, MAE, R² | Accuracy, F1-score, AUC |
| **Algoritmos comunes** | Regresión Lineal, SVR, Random Forest Regressor | Regresión Logística, SVM, Random Forest Classifier |

En resumen: **regresión → predice números; clasificación → predice categorías**.

---

**2. ¿Qué es la Regresión Lineal Simple y qué representa la ecuación ŷ = β₀ + β₁ · X?** Explica qué significan β₀  (intercepto) y β₁ (pendiente) y cómo interpretarlos en un contexto real. 

La **Regresión Lineal Simple** modela la relación entre una variable independiente (X) y una variable dependiente (y) mediante una línea recta.

**Ecuación:**

```
ŷ = β₀ + β₁ · X
```
| Símbolo | Nombre | Significado |
|---|---|---|
| **ŷ** | Valor predicho | El valor que el modelo estima para y |
| **β₀** | Intercepto | El valor de ŷ cuando X = 0 (punto donde la recta cruza el eje Y) |
| **β₁** | Pendiente | Cuánto cambia ŷ por cada unidad que aumenta X |
| **X** | Variable independiente | La variable de entrada (feature) |

**Interpretación en un contexto real:**

Supón que predices el **precio de un piso (en miles de €)** en función de sus **metros cuadrados**:

```
ŷ = 50 + 2.5 · X
```

- **β₀ = 50**: Un piso de 0 m² costaría 50.000 € (en la práctica, representa los costos base).
- **β₁ = 2.5**: Por cada metro cuadrado adicional, el precio sube 2.500 €.

El modelo aprende los valores óptimos de β₀ y β₁ minimizando el error entre las predicciones y los valores reales.

---

**3. ¿Cuáles son los supuestos de la Regresión Lineal?** Describe los supuestos principales: linealidad, homocedasticidad, independencia de errores y normalidad de residuos. ¿Por qué es importante verificarlos?

Para que los resultados sean válidos y fiables, la Regresión Lineal requiere verificar cuatro supuestos principales:

**1. Linealidad:**

La relación entre X e y debe ser **lineal** (una línea recta la representa bien). Si la relación es curvilínea, el modelo lineal dará predicciones pobres.

> **Cómo verificarlo:** Gráfico de dispersión de X vs y; si los puntos siguen una curva, el supuesto se viola.

**2. Homocedasticidad:**

La **varianza de los errores debe ser constante** para todos los valores de X. Lo contrario se llama heterocedasticidad.

> **Cómo verificarlo:** Gráfico de residuos vs valores predichos. Si la dispersión de los residuos aumenta o disminuye con el valor predicho, hay heterocedasticidad.

**3. Independencia de errores:**
Los errores (residuos) deben ser **independientes entre sí**; el error de una observación no debe estar relacionado con el de otra.

> **Cómo verificarlo:** Test de Durbin-Watson. Especialmente importante en series temporales.

**4. Normalidad de los residuos:**
Los residuos deben seguir una **distribución normal** (campana de Gauss), centrada en 0.

> **Cómo verificarlo:** Q-Q plot, histograma de residuos o test de Shapiro-Wilk.

**¿Por qué es importante verificarlos?**

Si los supuestos se violan, los coeficientes pueden ser sesgados, los intervalos de confianza incorrectos y las predicciones poco fiables. Verificarlos es la diferencia entre un modelo válido y uno engañoso.

---

**4. ¿Qué es la función de coste (Loss Function) en regresión?** Explica qué es el Error Cuadrático Medio (MSE) y el Error Absoluto Medio (MAE). ¿Cuál es la diferencia y cuándo preferirías usar uno u otro?

La **función de coste** (o función de pérdida) mide **cuánto se equivoca el modelo** comparando las predicciones (ŷ) con los valores reales (y). El objetivo del entrenamiento es **minimizar** esta función.

**MSE — Error Cuadrático Medio:**

```
MSE = (1/n) · Σ(yᵢ - ŷᵢ)²
```

- Eleva al cuadrado los errores → **penaliza fuertemente los errores grandes**.
- El resultado está en **unidades al cuadrado** (menos interpretable directamente).

**MAE — Error Absoluto Medio:**

```
MAE = (1/n) · Σ|yᵢ - ŷᵢ|
```

- Toma el valor absoluto → **trata todos los errores por igual**.
- El resultado está en **las mismas unidades que y** (más interpretable).

**¿Cuándo usar cada uno?**

| Situación | Métrica recomendada |
|---|---|
| Hay **outliers** y quiero que no dominen el modelo | **MAE** (más robusto) |
| Los **errores grandes son especialmente dañinos** | **MSE** (penaliza más) |
| Necesito una función **diferenciable** para optimización con gradiente | **MSE** (MAE no lo es en 0) |
| Quiero **interpretabilidad directa** en las unidades originales | **MAE** o **RMSE** |

---

**5. ¿Qué significa dividir los datos en train y test? ¿Por qué es necesario?** ¿Qué problema evitamos al no evaluar el modelo con los mismos datos con los que lo entrenamos? 

Dividir los datos en **train (entrenamiento)** y **test (prueba)** significa separar el conjunto de datos en dos partes:

- **Train set (~70-80%)**: El modelo aprende los patrones a partir de estos datos.
- **Test set (~20-30%)**: Se usa para evaluar el rendimiento del modelo en datos **que nunca ha visto**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**¿Por qué es necesario?**

Si evaluáramos el modelo con los **mismos datos de entrenamiento**, mediríamos cuánto "memorizó" los datos, no cuánto "aprendió" a generalizar. Esto da una falsa sensación de buen rendimiento.

**¿Qué problema evitamos?**

Evitamos el **overfitting (sobreajuste)**: el modelo aprende el "ruido" de los datos de entrenamiento en lugar de los patrones reales, y falla al predecir datos nuevos. El test set simula datos del mundo real.

---

**6. ¿Qué métricas se usan para evaluar un modelo de regresión?** Investiga y explica con sus fórmulas:

* R² (coeficiente de determinación)
* MSE (Mean Squared Error) 
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error) 

**R² — Coeficiente de Determinación:**

```
R² = 1 - [Σ(yᵢ - ŷᵢ)²] / [Σ(yᵢ - ȳ)²]
```

- Indica **qué proporción de la varianza** de y explica el modelo.
- Rango: de 0 a 1 (aunque puede ser negativo si el modelo es muy malo).
- **R² = 0.85** → El modelo explica el 85% de la varianza de y.
- **R² = 1** → Predicción perfecta. **R² = 0** → El modelo no explica nada.

**MSE — Mean Squared Error:**

```
MSE = (1/n) · Σ(yᵢ - ŷᵢ)²
```

- Promedio de los errores al cuadrado.
- Muy sensible a outliers. Unidades: **al cuadrado** de la variable objetivo.

**RMSE — Root Mean Squared Error:**

```
RMSE = √MSE = √[(1/n) · Σ(yᵢ - ŷᵢ)²]
```

- Raíz cuadrada del MSE → vuelve a las **unidades originales** de y.
- Más interpretable que el MSE y sigue penalizando errores grandes.
- **Es la métrica más usada en la práctica**.

**MAE — Mean Absolute Error:**

```
MAE = (1/n) · Σ|yᵢ - ŷᵢ|
```

- Promedio de los errores absolutos.
- Más robusto a outliers que MSE/RMSE.
- Fácil de interpretar: "en promedio, el modelo se equivoca en X unidades".

**Resumen comparativo:**

| Métrica | Unidades | Sensible a outliers | Interpretabilidad |
|---|---|---|---|
| R² | Sin unidades (0–1) | Moderada | Alta |
| MSE | Cuadráticas | Muy alta | Baja |
| RMSE | Originales | Alta | Alta |
| MAE | Originales | Baja | Muy alta |

---

**7. ¿Qué son los residuos y para qué sirve analizarlos?** ¿Qué indica un residuo cercano a 0? ¿Qué patrón en los residuos revelaría que el modelo no es adecuado?


---

**8. ¿Qué es el overfitting y cómo se detecta comparando R² en train vs. test?**

---

**9. ¿Qué es la validación cruzada (cross-validation) y qué ventaja ofrece frente a una sola división train/test?** 