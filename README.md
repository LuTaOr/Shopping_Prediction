"""
README - Predicción de Intención de Compra en un Comercio Online
"""

# **Descripción del Proyecto**
# Este proyecto busca predecir la intención de compra de usuarios en un comercio electrónico a partir de datos 
# de interacción con la plataforma web. Utilizando Machine Learning y Deep Learning, se desarrollaron modelos que 
# clasifican si un usuario realizará una compra (clase 1) o no (clase 0).
# 
# El objetivo principal es proporcionar a los responsables del comercio (como Manuel, el dueño del negocio en este caso) 
# información útil para implementar estrategias personalizadas de marketing y descuentos, optimizando recursos y 
# aumentando las conversiones.

# **Estructura del Repositorio**
# - `data/`  
#   Contiene el dataset original y los datasets preprocesados utilizados en el proyecto.
#   
# - `notebooks/`  
#   Notebooks con los análisis exploratorios, selección de características y entrenamiento de los modelos de 
#   Machine Learning y Deep Learning.
#   
# - `pipelines/`  
#   Código de los pipelines utilizados para el preprocesamiento de los datos, aplicación de técnicas de balanceo 
#   (como SMOTE) y entrenamientos.
# 
# - `model/`  
#   Modelos entrenados almacenados en formato `.pkl` para Machine Learning y `.keras` para Deep Learning.
# 
# - `images/`  
#   Imágenes y gráficas generadas durante el proyecto.
# 
# - `README.md`  
#   Este archivo de descripción del proyecto.

# **Tecnologías Utilizadas**
# - **Python 3.10**
#   - Librerías principales:
#     - `scikit-learn`
#     - `imbalanced-learn`
#     - `tensorflow`
#     - `matplotlib`
#     - `seaborn`
#     - `pandas`
#     - `numpy`

# **Pipeline del Proyecto**
# 1. **Carga de Datos:**  
#    Se utilizan datos sobre las interacciones de los usuarios en el comercio electrónico, con 18 características, 
#    incluyendo información administrativa, informativa y de productos.
# 
# 2. **Preprocesamiento:**  
#    - Transformación de variables categóricas con `OneHotEncoder`.
#    - Normalización de datos numéricos entre -1 y 1.
#    - Reducción de características irrelevantes.
# 
# 3. **Balanceo de Clases:**  
#    - Aplicación de SMOTE para equilibrar las clases desbalanceadas en el conjunto de entrenamiento.
# 
# 4. **Entrenamiento de Modelos:**  
#    - **Machine Learning:** Modelos como SVM, Random Forest, XGBoost y Logistic Regression.
#    - **Deep Learning:** Redes neuronales con múltiples capas densas y técnicas de regularización 
#      (BatchNormalization y Dropout).
# 
# 5. **Evaluación de Modelos:**  
#    - Métricas principales: **Balanced Accuracy**, **Recall de la clase 1** y **Confusion Matrix**.
#    - Se priorizó el Recall de la clase 1 para minimizar la cantidad de compradores potenciales mal clasificados.

# **Resultados**
# ### Modelos de Machine Learning
# - **Modelo seleccionado:** SVM con **SMOTE** aplicado.  
#   - Recall de la clase 1: **0.87**  
#   - Balanced Accuracy: **0.86**  
# 
# ### Modelos de Deep Learning
# - Arquitectura con tres capas densas y regularización.  
#   - Mejor Recall de la clase 1: **0.89**.  

# **Conclusiones**
# - La aplicación de técnicas como Feature Reduction y SMOTE mejora el Recall de la clase 1, permitiendo identificar 
#   más compradores potenciales.
# - El modelo de Deep Learning logró un rendimiento superior, pero a costa de un mayor tiempo de entrenamiento.
# - El sistema puede integrarse en producción para ofrecer promociones personalizadas en tiempo real.

# **Cómo Usar este Repositorio**
# 1. **Clonar el repositorio:**
#    ```bash
#    git clone https://github.com/tu_usuario/nombre_repositorio.git
#    cd nombre_repositorio
#    ```
# 
# 2. **Instalar dependencias:**
#    Asegúrate de tener un entorno virtual activo y ejecuta:
#    ```bash
#    pip install -r requirements.txt
#    ```
# 
# 3. **Ejecutar el pipeline:**
#    El pipeline se encuentra en el directorio `pipelines/` y automatiza el preprocesamiento y el entrenamiento:
#    ```bash
#    python pipelines/main_pipeline.py
#    ```
# 
# 4. **Probar los modelos guardados:**
#    Los modelos están disponibles en `model/`. Puedes cargarlos y hacer predicciones con el siguiente código:
#    ```python
#    import joblib
#    model = joblib.load('model/Best_params_model/Scoring_balanced_accuracy/SVM_best_model.pkl')
#    predictions = model.predict(X_test)
#    ```

# **Autor**
# Desarrollado por un apasionado de la Ciencia de Datos y el Machine Learning durante un bootcamp de Data Science.

# **Licencia**
# Este proyecto se encuentra bajo la licencia MIT.

