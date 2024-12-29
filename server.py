from flask import Flask, Response, render_template, json, request
from pymongo import MongoClient
import requests
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client['recomendacao']

data_movies = db["movies"]
data_credits = db["creditos"]

#########################################
try:
    mongo = MongoClient(
         host= "localhost", 
         port=27017,
         ServerSelectionTimeoutMS = 1000)
    db = mongo.recomendacao
    mongo.server_info()
except:
    print('ERRO - Cannot connect to db')
#########################################
# Retornar ID, CAPA e o NOME do FILME

@app.route("/", methods=["GET"])
def getMovies():
    try:
        # Obtém parâmetros de consulta (página e tamanho da página)
        search = request.args.get("search", "")  # Termo de busca (opcional)
        page = int(request.args.get("page", 1))  # Página atual (padrão: 1)
        per_page = int(request.args.get("per_page", 10))  # Itens por página (padrão: 20)

        # Calcula o número de documentos a serem ignorados
        skip = (page - 1) * per_page

        # Consulta com paginação
        movies_cursor = data_movies.find(
            { "title": { "$regex": f".*{search}.*", "$options": "i" } },
            { "title": 1, "poster_url": 1, "id": 1, "_id": 0 }
        ).skip(skip).limit(per_page)

        # Converte cursor para lista
        movies = list(movies_cursor)

        # Acrescentando as fotos por ID
        for movie in movies:
            movie["poster_url"] = getPosterMovies(movie["id"])

        # Conta o total de documentos que correspondem à consulta
        total_movies = data_movies.count_documents(
            { "title": { "$regex": f".*{search}.*", "$options": "i" } }
        )

        # Calcula o número total de páginas
        total_pages = (total_movies + per_page - 1) // per_page  # Arredondamento para cima

        start_page = max(1, page - 2)
        end_page = min(total_pages, page + 2)

        # Renderiza o template com os dados
        return render_template(
            "movies.html",
            movies_length=len(movies),
            movies=movies,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            total_movies=total_movies,
            start_page=start_page,
            end_page=end_page
        )
    except Exception as ex:
        return Response(
            response=json.dumps({"message": f"ERRO - Cannot connect to db: {ex}"}),
            status=500,
            mimetype="application/json",
        )
    
# ID, NOME DO FILME, CAPA, OVERVEIW, RECOMENDAÇÕES
@app.route("/movie-detail/<id>", methods=["GET"])
def getMovieByID (id):
    try:
        movie = data_movies.find_one({ "id": int(id)},
                                     { "id": 1, "title": 1, "overview": 1, "poster_url": 1, "_id": 0 })
        
        recommendedMovies = getRecomendedMovies(movie)  # Obtendo filmes recomendados
    
        # Acrescentando as fotos por ID
        if movie:
            # Imagem dos filmes
            movie['poster_url'] = getPosterMovies(movie['id'])
            
            index = 0
            while index < len(recommendedMovies):
                recommendedMovies[index]['poster_url'] = getPosterMovies(recommendedMovies[index]['movie_id'])
                index = index + 1

        print(recommendedMovies)
        return render_template(
            'movie-detail.html', 
            movie=movie,
            movies_length = len(recommendedMovies), 
            movies = recommendedMovies
        )
    except Exception as ex:
        return Response(response=json.dumps({ 'message': f'ERRO - Cannot connect to db: {ex}' }),status=500,mimetype='application/json')


                                    ##########################################
                                                    #CAPA
                                    ##########################################

def getPosterMovies(id):
     response = requests.get(f'https://api.themoviedb.org/3/movie/{id}?api_key=6cfc0d0aaa1109ec63d035a4d24369cc&language=en-US%22')
     data = response.json()
     return 'https://image.tmdb.org/t/p/w500/' +  data['poster_path']

                                    ##########################################
                                                    #CAPA
                                    ##########################################

def getRecomendedMovies(movie):
    all_records_movies = data_movies.find()
    all_records_credits = data_credits.find()

    list_records_movies = list(all_records_movies)
    list_records_credits = list(all_records_credits)

    movies = pd.DataFrame(list_records_movies)
    credits = pd.DataFrame(list_records_credits)

    movies = movies.drop("_id", axis= 1)
    credits = credits.drop("_id", axis = 1)

    def convert_json_to_str(obj):
        return json.dumps(obj)
    
    movies['genres'] = movies['genres'].apply(convert_json_to_str)
    movies['keywords'] = movies['keywords'].apply(convert_json_to_str)
    credits['cast'] = credits['cast'].apply(convert_json_to_str)
    credits['crew'] = credits['crew'].apply(convert_json_to_str)

    # Juntando as duas tabelas
    movies = movies.merge(credits, on = "title")

    movies = movies[['movie_id', 'genres', 'keywords', 'overview', 'title', 'id', 'cast', 'crew']]

    movies = movies.dropna()

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    def convert1(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter < 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L
    
    movies['cast'] = movies['cast'].apply(convert1)

    def convert2(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    
    movies['crew'] = movies['crew'].apply(convert2)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Transformando lista de string para lista de variaveis 
    movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

    movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]

    # junta os elementos da lista x em uma única sequência, separada por um espaço
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features = 5000, stop_words = 'english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    # similaridade entre os vetores
    similarity = cosine_similarity(vectors)

    sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x: x[1])[1:6]
    # Criando função para recomendar filmes
    # Criando função para recomendar filmes
    def recommend (movieTitle):
        movie_index = new_df[new_df['title'] == movieTitle].index[0]
        distences = similarity[movie_index]
        movies_list = sorted(list(enumerate(distences)), reverse = True, key = lambda x: x[1])[1:6]
        movies = [];

        for i in movies_list:
            movies.append(new_df.iloc[i[0]])
    
        return movies
    
    return recommend(movie['title'])

if __name__ == "__main__":
    app.run(port=80, debug= True)


