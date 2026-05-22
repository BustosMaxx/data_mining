## Informe - Entrega Parcial 3
Curso: Data Mining\
Alumno: Bustos, Maximiliano Miguel

En este trabajo se realiza un flujo completo de preparación de datos, detección de outliers, ingeniería de atributos, reducción de dimensionalidad, entrenamiento de modelo, y generación de una solución para subir a Kaggle.

### Pasos principales realizados en esta entrega

**A. Ingeniería de atributos**
   - Crea la columna `texto` uniendo `features` y `description` en minúsculas.
   - Se extrajo variables numéricas desde el texto: `c_dormitorios`, `c_baños`, `c_ambientes`, `superficie`, `precio_m2`, `precio_m2_total`. Esta última variable, es el producto entre el precio por m2 y la superficie.
   - Genera indicadores binarios para amenitis como `con_pileta`, `con_garage`, `con_ascensor`, `con_parque`, `a_estrenar`, etc.
   - Estandariza `property_type` reemplazando `casas` por `casa` y `departamentos` por `departamento`.
   - Se calculó `price_m2 = price / superficie`.
   - Agrupa por `property_type` y `location_2` para obtener el promedio de `price_m2` por barrio.
   - Se mezclo ese promedio de vuelta al dataset multiplicando el promedio con el valor de superficie de cada vivienda. De esta manera se envita el data leakeage `precio_total = mean(price_m2) * superficie`
   - Se probó transformar los datos a escala logaritmica para mejorar score, pero eso no tuvo un impacto, por lo que se desestimó.
   - Se generó variables dummy para `property_type`.
    con `get_dummies()`.


**B. Reducción de dimensionalidad y selección de variables**
   - Aplica `SelectKBest(chi2)`, correlación y RFE para explorar qué variables son más relevantes.
   - Utiliza `RandomForestRegressor` para seleccionar características.
   - Finalmente nos quedamos con 52 variables.

**C. Entrenamiento del modelo**
   - Se ajusta un `RandomForestRegressor` con `train_test_split`.
   - Se calculó errores RMSE en entrenamiento y prueba, reduciendo el error a `score_train=26242.74 - score_test=59730.95`
   - Se incluyó una sección opcional de *análisis de la importancia de variables*, dando como principales `c_baños` y `precio_total`

**D. Conclusión**
   
  - Se implementó un flujo integral que abarcó la preparación de datos, la detección de outliers, la ingeniería de atributos y la selección de variables mediante técnicas de reducción de dimensionalidad. Este proceso permitió obtener predicciones más precisas sobre el dataset objetivo. No obstante, un uso excesivo de la reducción de dimensionalidad o del filtrado de outliers puede derivar en un sobreajuste, incrementando el error en plataformas de evaluación como Kaggle. 
