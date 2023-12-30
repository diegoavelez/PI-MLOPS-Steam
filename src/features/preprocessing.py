# Importamos las librerías a usar
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_data(file_path):
    """
    Carga un dataset de juegos y realiza un preprocesamiento para su uso en sistemas de recomendación.

    Esta función lleva a cabo el preprocesamiento necesario para dos sistemas de recomendación diferentes:
    ítem-ítem y usuario-ítem. Para ítem-ítem, combina géneros y desarrolladores de juegos y utiliza
    TF-IDF para transformarlos en vectores numéricos, luego calcula la similitud del coseno entre los juegos.
    Para usuario-ítem, agrupa los datos por usuario y género y normaliza el tiempo total de juego,
    calculando luego la similitud del coseno entre los patrones de juego de los usuarios.

    Args:
        file_path (str): Ruta al archivo CSV que contiene los datos del juego.

    Returns:
        tuple: Contiene tres elementos en el siguiente orden:
            - DataFrame pandas con los datos del juego cargados.
            - Matriz de similitud del coseno para el sistema de recomendación ítem-ítem.
            - Matriz de similitud del coseno para el sistema de recomendación usuario-ítem.
    """
    
    df = pd.read_csv(file_path)
    
    # Eliminar duplicados basados en 'item_id'
    df_item = df.drop_duplicates(subset='item_id').copy()

    """
     Sistema de Recomendación Item - Item

    - El preprocesamiento del dataset se enfocará en el género y el desarrollador de cada uno de los registros

    - Esta es una elección del diseño del modelo basado en la relevancia de estas características para para determinar 
        la similitud entre los juegos.

    - *Género (Genre)*: Es una de las características más descriptivas y diferenciadoras de un juego. 
        Los usuarios a menudo tienen preferencias claras en cuanto a géneros, por lo que es un buen predictor de lo que podría gustarles.

    - *Desarrollador (Developer)*: Algunos jugadores son seguidores de ciertos desarrolladores y 
        tienden a disfrutar de otros juegos del mismo creador debido a un estilo, calidad o tema.

    - Al enfocarnos en un número limitado de características, el modelo puede ser más simple y eficiente, 
        mientras que proporciona recomendaciones útiles y precisas.

    - Ayuda también al problema de la "maldición de la dimensionalidad", donde demasiadas características pueden 
        hacer el análisis menos efectivo y más intensivo en computo.
    """
    
    # Preprocesamiento de datos para ítem-ítem
    
    # Combinamos los géneros y los desarrolladores de juegos en una sola cadena de texto
    df_item['combined_features'] = df_item['genres'] + " " + df_item['developer']
    
    # Utilizamos TF-IDF(Frecuencia de término - frecuencia inversa del documento) 
    # para convertir el texto en un conjunto de vectores numéricos
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_item['combined_features'])
    
    # Calculamos la similitud del coseno entre estos vectores para entender qué tan similares son los juegos entre sí
    cosine_sim_item = cosine_similarity(tfidf_matrix)

    """
    Sistema de Recomendación Usuario - Item

    - El preprocesamiento para este sistema se enfocará en el identificador del juego('item-id'), género('genre') y 
        tiempo de juego ('playtime_forever').

    - *Identificador de juego ('item-id')*: necesario para identificar cada juego individual en el dataset y relacionar 
        los juegos con los usuarios.

    - *Género ('genre')*: Uno de los factores más importantes que influyen en las preferencias de los usuarios. 
        Los usuarios a menudo tienen géneros favoritos y es probable que disfruten otros juegos dentro del mismo género. 

    - *Tiempo total de juego ('playtime_forever')*: esta medida proporciona cuánto tiempo un usuario ha dedicado a cada juego, lo cual puede ser un indicador fuerte de su preferencia, más que la reseña que es una acción opcional. Asume que los juegos en los que un usuario ha invertido más tiempo probablemente sean aquellos que más le gustan.

    - *Disponibilidad de Datos*: estas características pueden ser las más consistentemente disponibles y confiables en el conjunto de datos.

    - Estas tres características juntas permiten crear un perfil de preferencias para cada usuario basado en los tipos de juegos que juegan y cuánto tiempo pasan jugándolos. La idea es que si dos usuarios han dedicado cantidades de tiempo similares a géneros similares de juegos, es probable que tengan preferencias similares y, por lo tanto, podrían disfrutar de los mismos juegos.
    """
    
    """
    Debido a las limitaciones de memoria de la API, implementaremos un filtrado para los usuarios que más juegos tienen y
    llevan más tiempo total de horas jugadas, esto nos resultará en una muestra de los datos del dataframe original 
    para resolver el sistema de recomendación en un servidor virtual
    """
    
    # Calcular métricas clave
    user_metrics = df.groupby('user_id').agg(
        tiempo_total_jugado=pd.NamedAgg(column='playtime_forever', aggfunc='sum'),
        items_count=pd.NamedAgg(column='items_count', aggfunc='max')
    ).reset_index()

    # Ordenar usuarios por número de juegos jugados (items_count) y seleccionar los primeros 'max_usuarios'
    usuarios_seleccionados = user_metrics.sort_values(by='items_count', ascending=False).head(5000)

    # Obtener los IDs de los usuarios seleccionados
    usuarios_seleccionados_ids = usuarios_seleccionados['user_id']

    # Filtrar el DataFrame original para incluir solo los usuarios seleccionados
    df = df[df['user_id'].isin(usuarios_seleccionados_ids)]
    
    # Preprocesamiento de datos para usuario-Ítem
    
    # Agrupamos los datos por usuario y género y sumamos el tiempo total del juego por género
    user_genre_playtime = df.groupby(['user_id', 'genres'])['playtime_forever'].sum().unstack(fill_value=0)
    
    # Normalizamos estos datos para que cada fila sume 1, lo que nos da la proporción del tiempo dedicado a cada género
    # por usuario
    user_genre_playtime_normalized = user_genre_playtime.div(user_genre_playtime.sum(axis=1), axis=0)
    user_genre_playtime_normalized = user_genre_playtime_normalized.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Calculamos la similitud del coseno entre usuarios para entender qué tan similares son en términos de preferencia de juego
    user_similarity_user = cosine_similarity(user_genre_playtime_normalized)
    
    # Convirtiendo la matriz de similitud a un DataFrame
    user_similarity_df = pd.DataFrame(user_similarity_user, index=user_genre_playtime_normalized.index, columns=user_genre_playtime_normalized.index)


    return df, df_item, cosine_sim_item, user_similarity_df