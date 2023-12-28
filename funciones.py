import pandas as pd
import numpy as np

# Carga el dataset para los endpoints
file_path = './src/data/dataset_full.csv'
dataset = pd.read_csv(file_path)

def PlayTimeGenre_func(genre: str):
    """
    Función para encontrar el año con más horas jugadas para un género dado.

    Parameters:
    género (str): El género a analizar.

    Returns:
    int: El año con más horas jugadas para el género dado.
    float: El total de horas jugadas en ese año para el género dado.
    """
    # Filter data by the given genre
    filtered_data = dataset[dataset['genres'] == genre]

    # Group by release_year and sum playtime_forever
    playtime_by_year = filtered_data.groupby('release_year')['playtime_forever'].sum()

    # Find the year with the maximum playtime
    max_playtime_year = playtime_by_year.idxmax()
    max_playtime = playtime_by_year.max()

    # Construir y retornar el diccionario con el año y las horas de juego
    return {f"Año de lanzamiento con más horas jugadas para {genre}": max_playtime_year}

def UserForGenre_func(genre: str):
    """
    Función para encontrar el usuario con más horas jugadas y el acumulado de horas anual 
    un género dado.

    Parameters:
    género (str): El género a analizar.

    Returns:
    str: El ID del usuario con el mayor número de horas para el género dado.
    dict: Las horas de juego acumuladas por año para el género dado.
    """
    # Filter data by the given genre
    filtered_data = dataset[dataset['genres'] == genre]

    # Find the user with the most playtime
    user_playtime = filtered_data.groupby('user_id')['playtime_forever'].sum()
    top_user = user_playtime.idxmax()

    # Accumulate playtime by year
    playtime_by_year = filtered_data.groupby('release_year')['playtime_forever'].sum().to_dict()

    return {f"Usuario con más horas jugadas para Género {genre}": top_user, "Horas jugadas": playtime_by_year}

def usersRecommend_func(year: int):
    """
    Función para retornar el top 3 de los juegos más recomendados para un año dado.
    Un juego es considerado recomendado si tiene la columna sentiment con 1 (neutral) o 2 (positivo)

    Parameters:
    year (int): El año a analizar.

    Returns:
    list: Una lista de diccionarios con el top 3 de los juegos recomendados.
    """
    # Filter data by the given year and recommended sentiment (1 for neutral, 2 for positive)
    filtered_data = dataset[(dataset['release_year'] == year) & (dataset['sentiment'].isin([1, 2]))]

    # Count recommendations for each game
    recommended_games = filtered_data['app_name'].value_counts().head(3)

    # Preparing the result in the desired format
    top_3_games = [{"Puesto " + str(i + 1): game} for i, game in enumerate(recommended_games.index)]

    return top_3_games

def UsersWorstDeveloper_func(year: int):
    """
    Función para retornar el top 3 de los desarrolladores con menos juegos recomendados para un año dado
    Un juego es considerado no recomendado si la columna sentiment es cero (negativ0).

    Parameters:
    year (int): El año a analizar.

    Returns:
    list: Una lista de diccionarios con el top 3 de los desarrolladores con menos juegos recomendados.
    """
    # Filter data by the given year and not recommended sentiment (0 for negative)
    filtered_data = dataset[(dataset['release_year'] == year) & (dataset['sentiment'] == 0)]

    # Count not recommended games for each developer
    worst_developers = filtered_data['developer'].value_counts().head(3)

    # Preparing the result in the desired format
    top_3_developers = [{"Puesto " + str(i + 1): developer} for i, developer in enumerate(worst_developers.index)]

    return top_3_developers

def SentimentAnalysis_func(developer: str):
    """
    Function to perform sentiment analysis based on the developer name.
    It returns a dictionary with counts of user reviews categorized by sentiment.
    Treats null values in sentiment as 'Neutral' (1).

    Parameters:
    developer_name (str): The developer's name to analyze.

    Returns:
    dict: A dictionary with sentiment counts.
    """
    # Filter data by the given developer
    filtered_data = dataset[dataset['developer'] == developer]

    # Replace NaN values in sentiment with 1 (Neutral)
    filtered_data['sentiment'] = filtered_data['sentiment'].fillna(1)

    # Count reviews in each sentiment category
    sentiment_counts = filtered_data['sentiment'].value_counts()

    # Mapping sentiment codes to their meanings
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_results = {sentiment_mapping.get(k, k): v for k, v in sentiment_counts.items()}

    return {developer: sentiment_results}


    