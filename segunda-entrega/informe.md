# Entrega Parcial 2
Curso: Data Mining\
Alumno: Bustos, Maximiliano Miguel

En esta nueva entrega se realizó un análisis de outliers, generación de nuevos atributos a partir del dataset, y por último, imputación de los datos faltantes con nuevas técnicas aprendidas. Como resultado, se obtuvo una mejora en la predicción del algoritmo. 

## Filtrado
Previamente al análisis se realiza un filtrado de información, para que los valores de entrenamiento sean congruentes con los valores a predecir. 

## Generación de nuevos atributos
Atributo **"features":** Este campo contiene información extra de cada producto. Con el comando *extract* utilizando expresiones regulares, se pudo capturar la siguiente información: baños, dormitorios, superficie, garage.
```python
# dormitorios
df_ent.loc[:,"features"].str.extract(r"(\d+) dormitorios;|dormitorio: (\d+);|dormitorios:(\d+);")
# baños
df_ent.loc[:,"features"].str.extract(r"(\d+) baños;|bano: (\d+);|baños:(\d+);")
# garage
df_ent.loc[:,"features"].str.extract(r'(garage);|(cochera):|(cocheras):\d+')
# Superficie
df_ent.loc[:,"features"].str.extract(r";(\d+)\s*m²|scubierta: (\d+);| superficie total:(\d+)m.{1};|sup.\scubierta:(\d+)\s*m²| tcubierta:\s(\d*,\d*)\sm²| superficie\stotal:\s(\d+)")
```
## Outliers
Se observa outliers en el _precio_ de las viviendas. Para esto, se utilizó dos metodologías.\
Detección por **rango intercuartil**, el cual muestra una mayor cantidad de valores atípicos en el precio de los departamientos.\
Tambien se utilizó un gráfico de **Boxplot** para comparar el precio entre los barrios. En donde se observa _outliers_ en _location_2_: Palermo, Capital Federal, Buenos Aires, Villa Crespo y Villa Urquiza principalmente.
```python
df_ent.boxplot(by="location_2", column="price", ax=ax1)
```
## Imputación
Se realizó las siguientes imputaciónes:\
_Address_ NA se imputa con "sin info";\
_description_ NA se imputa con "sin info";\
*publication_date* reemplazo los NA con la moda "23 nov 2022";\
*publisher_id* NA reemplazo por la moda;\
*features*# Na imputo con "sin info";\
*location_* 3 y 4 con "sin info";\
*source* imputo con "sin info";\
Imputo cantidad de DORMITORIOS con la moda;\
Imputo cantidad BAÑOS con la moda;\
Imputo SUPERFICIE con el promedio;\
Garage es un booleano. Tiene o no tiene.

## Modelo de predicción
Dado que es un modelo de regresión, utiliza valores númericos para su predicción. Por lo que los nuevos atributos que se adicionaron al dataset, colaboran con el modelo. Y esto se evidencia tanto en el *"análisis de importancia de variables"* y en el resultado del score.

