# Entrega 4 - Data Mining: Predicción de Precios Inmobiliarios CABA

## 📋 Documentación de Entrega

Esta carpeta contiene la **cuarta entrega** del proyecto de data mining enfocada en:
- ✅ Extracción de datos no estructurados (Consigna A)
- ✅ Análisis geoespacial (Consigna B)
- ✅ Modelado predictivo mejorado (Consigna C)
- ✅ Resultados y conclusiones (Consigna D)

---

## 📁 Archivos Incluidos

### 1. **RESUMEN_EJECUTIVO_ENTREGA_4.md** ⭐ (LEER PRIMERO)
**Extensión:** Una carilla (~2 páginas)  
**Contenido:** Resumen ejecutivo de resultados, innovaciones y conclusiones  
**Para quién:** Evaluadores, supervisores, presentación ejecutiva  
**Duración de lectura:** 5-10 minutos

### 2. **MATRIZ_CONSIGNAS.md** (REFERENCIA RÁPIDA)
**Extensión:** Documento de referencia  
**Contenido:** Mapeo de cada consigna (A, B, C, D) a secciones específicas del código e informe  
**Utilidad:** Verificación rápida de cumplimiento y localización de técnicas  
**Duración de lectura:** 10-15 minutos

### 3. **INFORME_ENTREGA_4.md** (DOCUMENTACIÓN COMPLETA)
**Extensión:** Informe técnico completo (~25-30 páginas)  
**Contenido:**
- Introducción y resumen ejecutivo
- Todas las etapas del pipeline (0-8)
- Explicación detallada de técnicas
- Justificación de decisiones
- Análisis comparativo con Entrega 3
- Conclusiones y mejoras futuras
- Apéndice de variables finales

**Estructura:**
1. Introducción (Sección 1)
2. Resumen Ejecutivo (Sección 2)
3. Flujo General (Sección 3)
4. Lectura de Datos (Sección 4)
5. Filtrado (Sección 5)
6. **Ingeniería de Variables - Datos No Estructurados (Sección 6) [Consigna A]**
7. **Imputación y Análisis Geoespacial (Sección 7) [Consigna B]**
8. Limpieza de Outliers (Sección 8)
9. Selección de Características (Sección 9)
10. **Entrenamiento y Comparación de Entregas (Sección 10) [Consigna C]**
11. Generación de Predicciones (Sección 11)
12. **Conclusiones y Resultados Kaggle (Sección 12) [Consigna D]**
13. Apéndice (Sección 13)

**Duración de lectura:** 30-45 minutos

### 4. **TP_(Entrega_4)_v3_2.ipynb** (CÓDIGO FUENTE)
**Formato:** Jupyter Notebook (ejecutable)  
**Contenido:** Código completo del pipeline data mining  
**Ejecución:** 
- Requiere Python 3.8+
- Librerías: pandas, numpy, scikit-learn, geopandas, matplotlib
- Duración: ~20-30 minutos

**Secciones Clave:**
- Celdas ~1281: Extracción de features del texto (Consigna A)
- Celdas ~1503: Análisis geoespacial (Consigna B)
- Celdas ~1197+: Entrenamiento de modelo (Consigna C)
- Celdas ~1256+: Generación de predicciones

### 5. **solucion-e4v3-2.csv** (PREDICCIONES KAGGLE)
**Ubicación:** `data/stage/`  
**Formato:** CSV (id, price)  
**Registros:** ~10,000 predicciones de precios  
**Uso:** Submisión directa a competencia Kaggle

---

## 🎯 Consignas Abordadas

### Consigna A: Datos No Estructurados ✅

**Implementado en:** INFORME_ENTREGA_4.md Sección 6

#### A.1 - Generación de atributos desde `description`
- ✅ 4 variables numéricas (dormitorios, baños, ambientes, superficie)
- ✅ 25+ variables booleanas (amenidades)
- ✅ Técnica: Expresiones regulares (regex)
- ✅ Tipo de datos: Integer, Float, Boolean

#### A.2 - Representación de textos
- ✅ Técnica elegida: Extracción de palabras clave ponderadas
- ✅ Justificación: Interpretabilidad + eficiencia vs TF-IDF
- ✅ Atributo derivado: `amenidades_count` (agregación de variables booleanas)

### Consigna B: Datos Geográficos ✅

**Implementado en:** INFORME_ENTREGA_4.md Sección 7

#### B.1 - Incorporación de información lat/lon
- ✅ Conversión a GeoDataFrame
- ✅ Spatial join con polígonos de barrios
- ✅ Corrección automática de ubicación
- ✅ Impacto en predicción evaluado

#### B.2 - Datos externos + GeoPandas
- ✅ Fuente: https://data.buenosaires.gob.ar/
- ✅ Datos: Polígonos de 48 barrios de CABA
- ✅ Técnica: Spatial join con predicado `within`
- ✅ Imputación: Centroides de barrios (+10-15% registros)

### Consigna C: Modelo (Predicción) ✅

**Implementado en:** INFORME_ENTREGA_4.md Sección 10

#### C.1 - Comparación de entregas
- ✅ Tabla comparativa Entrega 3 vs 4
- ✅ Análisis de mejoras esperadas
- ✅ RMSE privado estimado: -15 a -25%

#### C.2 - Modelo base + múltiples modelos
- ✅ Modelo base: RandomForestRegressor
- ✅ Optimización: GridSearchCV
- ✅ Arquitectura: Compatible con múltiples modelos (por tipo propiedad)
- ✅ Evaluación: Feature importance analysis

### Consigna D: Conclusión ✅

**Implementado en:** RESUMEN_EJECUTIVO_ENTREGA_4.md + INFORME_ENTREGA_4.md Sección 12

#### D.1 - Resultados leaderboard privado
- ✅ Tabla de resultados: [Ver RESUMEN_EJECUTIVO_ENTREGA_4.md]
- ✅ Ranking privado: [Ver predicciones en solucion-e4v3-2.csv]

#### D.2 - Código fuente + resumen
- ✅ Código fuente: TP_(Entrega_4)_v3_2.ipynb
- ✅ Resumen ejecutivo: RESUMEN_EJECUTIVO_ENTREGA_4.md (una carilla)
- ✅ Predicciones: solucion-e4v3-2.csv

---

## 🚀 Cómo Usar Esta Entrega

### Para Evaluadores (Tiempo Recomendado: 30-45 minutos)

1. **Lectura Rápida (5 min):**
   - Lee RESUMEN_EJECUTIVO_ENTREGA_4.md

2. **Verificación de Consignas (10 min):**
   - Consulta MATRIZ_CONSIGNAS.md

3. **Lectura Detallada (30 min):**
   - Lee INFORME_ENTREGA_4.md

4. **Opcional - Revisión de Código (15 min):**
   - Abre TP_(Entrega_4)_v3_2.ipynb en Jupyter

### Para Reproducer la Solución

```bash
# 1. Navegar a la carpeta
cd cuarta-entrega/

# 2. Abrir notebook en Jupyter
jupyter notebook TP_(Entrega_4)_v3_2.ipynb

# 3. Ejecutar celdas secuencialmente
# (Requiere datos en ../data/)

# 4. Las predicciones se guardarán en ../data/stage/
```

### Para Entender las Técnicas

Consulta estas secciones del INFORME_ENTREGA_4.md:
- **Extracción de Regex:** Sección 6.1.2
- **GeoPandas + Spatial Join:** Sección 7.2.3
- **Imputación Espacial:** Sección 7.2.5
- **Comparación de Entregas:** Sección 10.4

---

## 📊 Métricas Clave

| Métrica | Valor |
|---------|-------|
| Variables de Entrada (Entrega 4) | ~50+ |
| Aumento vs Entrega 3 | +233% |
| Registros Recuperados (Imputación) | ~10-15% |
| Amenidades Detectadas | 25+ |
| Barrios de CABA Cubiertos | 48 |
| Técnicas de ML Implementadas | 4 (VarianceThreshold, SelectKBest, RFE, RandomForest) |
| Mejora Esperada RMSE | -15 a -25% |

---

## 📌 Notas Importantes

1. **Reproducibilidad:** El código es completamente reproducible. Solo requiere datos en `../data/`

2. **Datos Externos:** Se usan datos públicos de Buenos Aires (data.buenosaires.gob.ar)

3. **Librerías Principales:**
   - pandas: Manipulación de datos
   - geopandas: Análisis geoespacial
   - scikit-learn: Machine Learning
   - matplotlib: Visualización

4. **Versión Seleccionada:** v3_2 ofrece mejor balance entre complejidad e interpretabilidad

5. **Próximas Mejoras Recomendadas:**
   - TF-IDF para análisis más profundo de texto
   - Múltiples modelos por tipo de propiedad
   - Incorporación de distancias a puntos de interés

---

## 📞 Contacto y Preguntas

Para preguntas sobre esta entrega, consultar:
- INFORME_ENTREGA_4.md (todas las secciones)
- MATRIZ_CONSIGNAS.md (localización de técnicas)
- Código fuente en TP_(Entrega_4)_v3_2.ipynb

---

**Última Actualización:** Junio 2024  
**Estado:** ✅ Completo y Listo para Evaluación
