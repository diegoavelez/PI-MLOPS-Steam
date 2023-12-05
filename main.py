from fastapi import FastAPI
from funciones import *
import pandas as pd

app = FastAPI()

@app.get('/')
def Presentacion():
    return {'Proyecto de MLOPS usando datos de la plataforma STEAM, las funciones implementdas se acceden en /docs'}

@app.get("/playtime-genre/{genre}")
def PlayTimeGenre(genre: str):
    """
    Función para encontrar el año con más horas jugadas para un género dado.

    Parameters:
    género (str): El género a analizar.

    Returns:
    int: El año con más horas jugadas para el género dado.
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
    género (str): El género a analizar.

    Returns:
    str: El ID del usuario con el mayor número de horas para el género dado.
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
    year (int): El año a analizar.

    Returns:
    list: Una lista de diccionarios con el top 3 de los juegos recomendados.
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
    year (int): El año a analizar.

    Returns:
    list: Una lista de diccionarios con el top 3 de los desarrolladores con menos juegos recomendados.
    """
    try:
        return UsersWorstDeveloper_func(year)
    except Exception as e:
        return {"Error":int(e)}
    

@app.get("/sentiment-analysis/{year}")
def SentimentAnalysis(developer: str):
    """
    Function to perform sentiment analysis based on the developer name.
    It returns a dictionary with counts of user reviews categorized by sentiment.
    Treats null values in sentiment as 'Neutral' (1).

    Parameters:
    developer_name (str): The developer's name to analyze.

    Returns:
    dict: A dictionary with sentiment counts.
    """
    try:
        return SentimentAnalysis_func(developer)
    except Exception as e:
        return {"Error":str(e)}
