# Entrega 4
## Predicción de Precios Inmobiliarios en CABA
**Alumno: Bustos Maximiliano**

### Contexto
La cuarta entrega del proyecto de data mining se enfocó en elevar significativamente la capacidad predictiva del modelo mediante la incorporación de **datos no estructurados** (descripciones de propiedades) y **análisis geoespacial avanzado** (datos públicos de Buenos Aires).

### Puntos principales

#### 1. Extracción Inteligente de Datos del Texto
Se desarrolló un sistema de extracción de features basado en expresiones regulares que procesa los campos `description` y `features` de cada propiedad:
- **Extracción numérica:** 4 variables (dormitorios, baños, ambientes, superficie)
- **Indicadores de amenidades:** 25+ variables booleanas (pileta, ascensor, jardín, seguridad, ascensor, etc.)
- **Text mining:** En un notebook separado se aplico esta técnica y se extrajo 7 variables booleanas ("con_suite","con_amenities","con_dependencia","monoambiente","palier_privado","palier_privado","con_lavadero") de los `Feature importance analysis`

#### 2. Análisis Geoespacial con Datos Abiertos
Integración de la librería GeoPandas con polígonos de barrios del portal de datos abiertos de Buenos Aires (data.buenosaires.gob.ar):
- **Spatial join:** Corrección automática de barrios asignando el polígono correcto según coordenadas
- **Imputación espacial:** Recuperación de registros sin coordenadas usando centroides de barrios
- **Resultado:** Aumento en volumen de datos de entrenamiento y mejora en precisión geográfica

#### 3. Modelo 
RandomForestRegressor optimizado iterenado entre diferentes parámetros, quedando n_estimators=300 -- max_depth=10, que comparando explícitamente con Entrega 3, redujo el overfiting y redujo el error en test.

### Resultados en Kaggle

**Versión Seleccionada:** `TP_(Entrega_4)_v3_2.ipynb`  
**Archivo de Submisión:** `solucion-e4v3-4.csv`

### Conclusión

La cuarta entrega logró una **mejora significativa en la capacidad predictiva** del modelo mediante la incorporación sistemática de datos no estructurados y análisis geoespacial. Aumento de variables predictoras, combinado con la recuperación de datos y calibración geográfica, establece una **base sólida para futuras iteraciones** del proyecto.


