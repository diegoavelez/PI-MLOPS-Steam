
<p align="center"><img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"></p>

<h1 align='center'> Proyecto Individual N°1</h1>

<h2 align='center'> Machine Learning Operations (MLOps)</h2>

<h2 align='center'>Diego Alejandro Vélez, DATAPT05</h2>

Modelo de ML para crear un modelo de recomendación de videojuegos para usuarios


## **`Tabla de Contenidos`**

- [Introducción](#introducción)
- [Diccionario de datos](##diccionario)
- [Organización del proyecto](#organizacion)
- [Desarrollo](#desarrollo)
    - [Extracción y limpieza](#extracción-y-limpieza-de-datos)
    - [Preparación dataset](#preparacion-dataset)
    - [Construcción de funciones](#construcción-de-funciones)
    - [Análisis exploratorio de los datos](#análisis-exploratorio-de-los-datos-eda)
    - [Construcción de modelos de recomendación](#construcción-de-modelos-de-recomendación)
    - [Puesta en marcha](#puesta-en-marcha)


# Introducción

Este proyecto de Machine Learning contempla el ciclo de vida de un proyecto habitual desde el tratamiento y recolección de los datos (Data Engineer Stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos dato.

El proyecto en concreto pretende llevar a cabo un estudio basado en Machine Learning Operations (MLOps). Este estudio tiene la siguiente propuesta de trabajo:

<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png"  height=500>
</p>

1. **Extracción y limpieza de los datos:** se realizará un extracción de los registros desde archivos json para ser convertidos a datasets que puedan ser manipulados para las consultas finales.

2. **Preparación del dataset:** se prepará un dataset final para las consultas usando los datasets obtenidos anteriormente.

3. **Construcción de funciones:** se desarrollará un script para las que las funciones cumplan con las consultas requeridas en la propuesta de trabajo.

4. **Análisis Exploratorio de los Datos**: posteriormente para construir los modelos de recomendación se hará un análisis exploratorio de los datos, buscando la información más relevante a la hora de analizar y entender cómo están distribuidos nuestros datos.

5. **Construcción de modelos**: se construyeron modelos de recomendación, usando similitud coseno tanto del tipo dado un item, te recomiendo cinco items parecidos; como también se implementó recomendaciones basadas en un usuario, entregando un usuario te recomiendo juegos de usuarios similares.

6. **Puesta en marcha**: se construyó un main.py invocando las funciones correspondientes a las consultas del proyecto y el sistema de recomendación.

## Diccionario de datos

<p align="center"><img src="./references/Diccionario de datos.png"></p>

# Organización del proyecto
------------

├── Makefile           <- Makefile con comandos como `make data` o `make train` para futuras implementaciones si es necesario
├── README.md          <- El README de nivel superior para desarrolladores que usan este proyecto.
├── datos
│   ├── processed     <- Los conjuntos de datos finales y canónicos para el modelado.
│   └── raw           <- El volcado de datos original e inmutable.
│
├── notebooks         <- Notebooks Jupyter. La convención de nombres es un número (para el orden),
│                         las iniciales del creador, y una descripción corta delimitada por `-`, por ejemplo,
│                         `1.0-jqp-exploración-inicial-de-datos`.
│
├── references         <- Diccionarios de datos, manuales y todos los demás materiales explicativos.
│
│
├── requirements.txt   <- El archivo de requisitos para reproducir el entorno de análisis, por ejemplo,
│                         generado con `pip freeze > requirements.txt`
│
├── setup.py           <- hace que el proyecto sea instalable con pip (pip install -e .) para que src pueda ser importado
├── src                <- Código fuente para usar en este proyecto.
│   ├── __init__.py    <- Convierte src en un módulo de Python
│   │
│   ├── datos          <- Datasets definitivos para usar para la API del proyecto
│   │   └── dataset_full.csv
│   │
│   ├── features       <- Scripts para convertir datos crudos en características para el modelado
│   │   └── preprocessing.py
│   │
│   ├── models         <- Scripts para entrenar modelos y luego usar modelos entrenados para hacer
│       │                 predicciones
│       └── modelos.py
│   
│
└── tox.ini            <- archivo tox con configuraciones para ejecutar tox; consulta tox.readthedocs.io

# Desarrollo del proyecto

## Extracción y limpieza de los datos

- A partir de los 3 dataset proporcionados (steam_games, user_reviews y user_items) referentes a la plataforma de Steam, se realiza una extracción desde los archivos json para ser transformados a datasets que podamos usar en el resto del proyecto y resolver la propuesta de trabajo.

### steam_games.json

- Se hizo lectura correcta del archivo para transformar a dataframe
- Limpieza de los valores nulos en la columna 'genre'
- Se creo una columna con el año de lanzamiento a partir de la fecha de lanzamiento
- Se hizo un explode de la columna 'genre' para que quedara cada registro con un solo género
- Se eliminaron columnas que no serán necesarias para las consultas.
- Se eliminaron filas para registros que tuvieran valores nulos en 'genres', 'release year', 'developer' 

### user_reviews.json

- Definición de funciones para la lectura de caracteres especiales y de escape
- Lectura exitosa del archivo y conversión a dataframe
- Se realizó un ingeniería de atributos para obtener la columna sentiment a partir de un análisis de sentimiento de la columna review
- Se eliminaron columnas que son innecesarias para el proyecto

### user.items.json

- Funciones para la lectura de registros con diccionarios anidados y que fueron convertidos a filas individuales
- Eliminación de columnas innecesarias para el proyecto

## Preparación Dataset

- Se realizó un merge solamente con los juegos que tienen reseñas para un mejor análisis de las consultas y para optimización de la memoria
- Se gestionaron los valores nulos, eliminando aquellos registros donde genres, release_year y playtime_forever eran nulos.
- En el caso de la columna 'price' los nulos se convirtieron a 0, y algunas cadenas de string que indicaban que era gratis (free) se actualizó a precio 0.
- Se eliminaron los duplicados del dataset resultante

## Construcción de funciones

Se construyeron las funciones con el siguiente criterio de la propuesta de trabajo:

+ def **PlayTimeGenre( *`genero` : str* )**:
    Debe devolver `año` con mas horas jugadas para dicho género.
  
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

+ def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
			     "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

+ def **UsersRecommend( *`año` : int* )**:
   Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **UsersWorstDeveloper( *`año` : int* )**:
   Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **sentiment_analysis( *`empresa desarrolladora` : str* )**:
    Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor. 

Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}

## Análisis Exploratorio de los datos (EDA)

- Se hicieron estadísticas descriptivas básicas de las columnas numéricas del dataset resultante
- Se realizó un análisis de outliers para las columnas playtime_forever y price.
- Se realizó un análisis de las variables categóricas
- Se realizó una nube de palabras para los títulos de los juegos
- Se creó una matriz de correlación entre las variables numéricas para entender las relaciones entre sí.
- Se elaboró un análisis de los géneros de los juegos, desarrolladores y de los años de lanzamientos más relevantes.

## Construcción de modelos de recomendación

- Preprocesamiento necesario para implementar los dos modelos de recomendación
- Se crea una función para realizar dicho preprocesamiento para los modelos ítem-ítem y usuario-ítem.
- Para ítem-ítem, combina géneros y desarrolladores de juegos y utiliza TF-IDF para transformarlos en vectores numéricos, luego calcula la similitud del coseno entre los juegos.
- Para usuario-ítem, agrupa los datos por usuario y género y normaliza el tiempo total de juego, calculando luego la similitud del coseno entre los patrones de juego de los usuarios.

## Puesta en marcha

- Se crean funciones para invocar los modelos de recomendación desde teniendo en cuenta las matrices de similitud necesarias para calcular las recomendaciones
- Se implementaron los dos modelos de recomendación que pueden ser invocadas desde la API, recomendación item-item, usuario-item.
- En el archivo main.py se invocan todas las funciones necesarias para la propuesta de trabajo y puedan ser consumidas desde la API.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
