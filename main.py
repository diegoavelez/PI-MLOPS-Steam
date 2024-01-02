import sys
from fastapi import FastAPI
from funciones import *
import pandas as pd

# Añadir el directorio de los módulos a sys.path
sys.path.append("./src/models")
sys.path.append("./src/features")

from preprocessing import load_and_preprocess_data
from modelos import recomendacion_juego, recomendacion_usuario

app = FastAPI()

@app.get('/')
def Presentacion():
    return {'Proyecto de MLOPS usando datos de la plataforma STEAM, las funciones implementdas se acceden en /docs'}

@app.get("/playtime-genre/{genre}")
def PlayTimeGenre(genre: str):
    """
    Función para encontrar el año con más horas jugadas para un género dado.

    Parameters:
    género (str): el género a analizar.

    Returns:
    int: el año con más horas jugadas para el género dado.
    float: El total de horas jugadas en ese año para el género dado.
    """
    
    try:
        return PlayTimeGenre_func(genre)
    except Exception as e:
        return {"Error":str(e)}

@app.get("/user-for-genre/{genre}")
def UserForGenre(genre: str):
    """
    Función para encontrar el usuario con más horas jugadas y el acumulado de horas anual 
    un género dado.

    Parameters:
    género (str): el género a analizar.

    Returns:
    str: el ID del usuario con el mayor número de horas para el género dado.
    dict: Las horas de juego acumuladas por año para el género dado.
    """
    try:
        return UserForGenre_func(genre)
    except Exception as e:
        return {"Error":str(e)}

@app.get("/users-recommend/{year}")
def usersRecommend(year: int):
    """"
    Función para retornar el top 3 de los juegos más recomendados para un año dado.
    Un juego es considerado recomendado si tiene la columna sentiment con 1 (neutral) o 2 (positivo)

    Parameters:
    year (int): el año a analizar.

    Returns:
    list: una lista de diccionarios con el top 3 de los juegos recomendados.
    """
    try:
        return usersRecommend_func(year)
    except Exception as e:
        return {"Error":int(e)}

@app.get("/users-worst-developer/{year}")
def UsersWorstDeveloper(year: int):
    """
    Función para retornar el top 3 de los desarrolladores con menos juegos recomendados para un año dado
    Un juego es considerado no recomendado si la columna sentiment es cero (negativ0).

    Parameters:
    year (int): el año a analizar.

    Returns:
    list: una lista de diccionarios con el top 3 de los desarrolladores con menos juegos recomendados.
    """
    try:
        return UsersWorstDeveloper_func(year)
    except Exception as e:
        return {"Error":int(e)}
    

@app.get("/sentiment-analysis/{developer}")
def SentimentAnalysis(developer: str):
    """
    Función para obtener el análisis de sentimiento basado en el nombre del desarollador.
    Retorna un diccionario con los valores correspondientes a cada tipo de reseña, positiva, negativa o neutral.
    Trata los valores nulos en la columna sentimiento como neutrales (1).

    Parameters:
    Nombre del desarrollador (str): el nombre del desarrollador a analizar.

    Returns:
    dict: un diccionario con el conteo de valores para el análisis de sentimiento.
    """
    try:
        return SentimentAnalysis_func(developer)
    except Exception as e:
        return {"Error":str(e)}
    
# Cargar y preprocesar los datos
file_path = './src/data/dataset_full.csv'
df, df_item, cosine_sim_item, user_similarity_df = load_and_preprocess_data(file_path)

# Endpoints de la API
@app.get("/recomendacion-item/{item_id}")
async def recomendacion_por_item(item_id: int):
    """
    Genera una lista de juegos recomendados similares a un juego específico.

    Esta función identifica juegos similares a partir de un juego dado, utilizando la similitud del coseno 
    basada en características combinadas como géneros y desarrolladores. La función devuelve los cinco 
    juegos más similares, excluyendo el juego de entrada.

    Parameters:
        item_id (int): el ID del juego para el cual se harán las recomendaciones.
        df (pd.DataFrame): El DataFrame que contiene los datos de los juegos, incluyendo 'item_id', 'genres', y 'developer'.
        cosine_sim (numpy.ndarray): Matriz de similitud del coseno precalculada para los juegos.

    Returns:
        list of dict: una lista de diccionarios, donde cada diccionario contiene 'item_id' y 'app_name' 
        de los juegos recomendados. Devuelve una lista vacía si el juego no se encuentra en el dataset.
    
    Ejemplo:
        recomendaciones = recomendacion_juego(123, df, cosine_sim)
        # Esto podría devolver juegos similares al juego con ID 123.
    """
    recomendaciones = recomendacion_juego(item_id, df_item, cosine_sim_item)
    return {"item_id": item_id, "recomendaciones": recomendaciones}

@app.get("/recomendacion-usuario/{user_id}")
async def recomendacion_por_usuario(user_id: str):
    """
    Genera recomendaciones de juegos para un usuario específico basándose en usuarios similares.

    Esta función busca usuarios con patrones de juego similares al usuario dado y recomienda juegos que 
    estos usuarios similares han jugado, pero que el usuario en cuestión aún no ha probado. Utiliza la 
    matriz de similitud del coseno entre usuarios para determinar qué usuarios son similares.

    Parameters:
        user_id (int): el ID del usuario para el cual se realizará la recomendación.
        df (pd.DataFrame): El DataFrame que contiene los datos de los juegos y usuarios.
        user_similarity_df (pd.DataFrame): DataFrame que representa la matriz de similitud del coseno entre usuarios.
        num_recommendations (int, opcional): Número de recomendaciones a generar. Por defecto es 5.

    Returns:
        list of dict: una lista de diccionarios, donde cada diccionario contiene 'item_id' y 'app_name' 
                      de los juegos recomendados. Devuelve una lista vacía si el usuario no se encuentra en el dataset.

    Ejemplo:
        recomendaciones_usuario = recomendacion_usuario_mod(456, df, user_similarity_df)
        # Esto podría devolver juegos recomendados para el usuario con ID 456.
    """
    recomendaciones = recomendacion_usuario(user_id, df, user_similarity_df)
    return {"user_id": user_id, "recomendaciones": recomendaciones}
