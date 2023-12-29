# Importar las librerías a usar
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Función para recomendación ítem-ítem
def recomendacion_juego(item_id, df, cosine_sim):
    """
    Genera una lista de juegos recomendados similares a un juego específico.

    Esta función identifica juegos similares a partir de un juego dado, utilizando la similitud del coseno 
    basada en características combinadas como géneros y desarrolladores. La función devuelve los cinco 
    juegos más similares, excluyendo el juego de entrada.

    Args:
        item_id (int): El ID del juego para el cual se harán las recomendaciones.
        df (pd.DataFrame): El DataFrame que contiene los datos de los juegos, incluyendo 'item_id', 'genres', y 'developer'.
        cosine_sim (numpy.ndarray): Matriz de similitud del coseno precalculada para los juegos.

    Returns:
        list of dict: Una lista de diccionarios, donde cada diccionario contiene 'item_id' y 'app_name' 
                      de los juegos recomendados. Devuelve una lista vacía si el juego no se encuentra en el dataset.
    
    Ejemplo:
        recomendaciones = recomendacion_juego(123, df, cosine_sim)
        # Esto podría devolver juegos similares al juego con ID 123.
    """
    
    # Si el juego no está en el DataFrame, devuelve un mensaje de error.
    if item_id not in df['item_id'].values:
        return "El juego con el ID proporcionado no se encuentra en el dataset."
    
    # Busca el índice del juego en el DataFrame usando el ID proporcionado.
    idx = df.index[df['item_id'] == item_id].tolist()[0]
    
    # Crea una lista de pares (índice, puntuación de similitud) para todos los juegos, 
    # basándose en la fila correspondiente al juego en la matriz de similitud.
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordena los juegos de acuerdo a su puntuación de similitud, de mayor a menor.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Selecciona los primeros 5 juegos (excluyendo el propio juego que es el más similar a sí mismo).
    sim_scores = sim_scores[1:6]  # El primero es el propio juego
    
    # Extrae los índices de los juegos recomendados de los pares (índice, puntuación).
    game_indices = [i[0] for i in sim_scores]
    
    # Utiliza los índices para obtener los IDs y nombres de los juegos recomendados del DataFrame y los devuelve.
    recomendaciones = df.iloc[game_indices][['item_id', 'app_name']].to_dict('records')
    return recomendaciones

# Función para recomendación usuario-item
def recomendacion_usuario(user_id, df, user_similarity_df, num_recommendations=5):
    """
    Genera recomendaciones de juegos para un usuario específico basándose en usuarios similares.

    Esta función busca usuarios con patrones de juego similares al usuario dado y recomienda juegos que 
    estos usuarios similares han jugado, pero que el usuario en cuestión aún no ha probado. Utiliza la 
    matriz de similitud del coseno entre usuarios para determinar qué usuarios son similares.

    Args:
        user_id (int): El ID del usuario para el cual se realizará la recomendación.
        df (pd.DataFrame): El DataFrame que contiene los datos de los juegos y usuarios.
        user_similarity_df (pd.DataFrame): DataFrame que representa la matriz de similitud del coseno entre usuarios.
        num_recommendations (int, opcional): Número de recomendaciones a generar. Por defecto es 5.

    Returns:
        list of dict: Una lista de diccionarios, donde cada diccionario contiene 'item_id' y 'app_name' 
                      de los juegos recomendados. Devuelve una lista vacía si el usuario no se encuentra en el dataset.

    Ejemplo:
        recomendaciones_usuario = recomendacion_usuario_mod(456, df, user_similarity_df)
        # Esto podría devolver juegos recomendados para el usuario con ID 456.
    """
    
    # Si el usuario no está en el DataFrame, devuelve un mensaje de error.
    if user_id not in user_similarity_df.index:
        return "El usuario con el ID proporcionado no se encuentra en el dataset."

    # Ordena a los usuarios en función de su similitud con el usuario objetivo y selecciona los más similares
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    # Crea un conjunto de IDs de juegos que el usuario objetivo ya ha jugado.
    user_games = set(df[df['user_id'] == user_id]['item_id'])
    
    # Recorre los usuarios similares, recopilando juegos que ellos han jugado pero el usuario objetivo no. 
    # Detiene el bucle una vez que se alcanza el número deseado de recomendaciones.
    recommended_games = set()
    for similar_user in similar_users:
        similar_user_games = set(df[df['user_id'] == similar_user]['item_id'])
        new_recommendations = similar_user_games.difference(user_games)
        
        # Actualiza el conjunto de juegos recomendados, pero solo hasta alcanzar num_recommendations
        recommended_games.update(new_recommendations)
        if len(recommended_games) >= num_recommendations:
            break
        
    # Se asegura que sólo devuelva exactamente el número de juegos recomendados
    final_recommendations = list(recommended_games)[:num_recommendations]
    recomendaciones = df[df['item_id'].isin(final_recommendations)].drop_duplicates(subset='item_id')[['item_id', 'app_name']].to_dict('records')

    return recomendaciones