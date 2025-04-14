
import math
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['Genre'] + ' ' + x['Tags'] + ' ' + x['Actors'] + ' ' + x['ViewerRating']

def get_recommendations(title, cosine_sim):
    global result
    title = title.replace(' ', '').lower()
    idx = indices.get(title)
    if idx is not None:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:51]  # Get top 50 recommendations
        movie_indices = [i[0] for i in sim_scores]
        result = netflix_data.iloc[movie_indices]
        result.reset_index(inplace=True)
        return result
    else:
        print(f"Title '{title}' not found in indices.")
        return pd.DataFrame()  # Return empty DataFrame if title not found

# Loading and preparing the dataset
netflix_data = pd.read_csv('NetflixDataset.csv', encoding='latin-1', index_col='Title')
netflix_data.index = netflix_data.index.str.title()
netflix_data = netflix_data[~netflix_data.index.duplicated()]
netflix_data.rename(columns={'View Rating': 'ViewerRating'}, inplace=True)

# Extracting features
Language = netflix_data.Languages.str.get_dummies(',')
Lang = Language.columns.str.strip().values.tolist()
Titles = list(set(netflix_data.index.to_list()))

# Preprocessing the data
netflix_data['Genre'] = netflix_data['Genre'].astype('str')
netflix_data['Tags'] = netflix_data['Tags'].astype('str')
netflix_data['IMDb Score'] = pd.to_numeric(netflix_data['IMDb Score'], errors='coerce').fillna(6.6)
netflix_data['Actors'] = netflix_data['Actors'].astype('str')
netflix_data['ViewerRating'] = netflix_data['ViewerRating'].astype('str')
new_features = ['Genre', 'Tags', 'Actors', 'ViewerRating']
selected_data = netflix_data[new_features]


for new_feature in new_features:
    selected_data.loc[:, new_feature] = selected_data.loc[:, new_feature].apply(prepare_data)

selected_data.index = selected_data.index.str.lower().str.replace(" ", '')
selected_data.loc[:, 'soup'] = selected_data.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(selected_data['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
selected_data.reset_index(inplace=True)
indices = pd.Series(selected_data.index, index=selected_data['Title'])

app = Flask(__name__)

@app.template_filter('zip')
def zip_filter(*args):
    return zip(*args)

@app.route('/')
def index():
    return render_template('index.html', languages=Lang, titles=Titles)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower()
    type = request.args.get('type')

    if type == 'title':
        suggestions = [title for title in Titles if query in title.lower()]
    elif type == 'language':
        suggestions = [language for language in Lang if query in language.lower()]
    else:
        suggestions = []

    return jsonify(suggestions)

@app.route('/about', methods=['POST'])
def getvalue():
    global result
    result = pd.DataFrame() 
    movienames = request.form.getlist('titles')  # This retrieves multiple titles
    languages = request.form.getlist('languages')  # This retrieves multiple languages
 
    print("Selected Movies:", movienames)
    print("Selected Languages:", languages)

    for moviename in movienames:
        # recommendations for each movie title chose
        get_recommendations(moviename, cosine_sim2)
        for language in languages:
            if result.empty:
                print("No recommendations found for this movie.")
                continue  # Skip if no recommendations exist

            # Filter recommendations based on selected languages
            filtered = result[result['Languages'].str.contains(language, na=False)]
            print(f"Filtering by language '{language}': Found {filtered.shape[0]} matches.")

            result = pd.concat([filtered, result], ignore_index=True) 

    result.drop_duplicates(keep='first', inplace=True)

    print("Result DataFrame after filtering:", result)

    if 'IMDb Score' in result.columns:
        result.sort_values(by='IMDb Score', ascending=False, inplace=True)
    else:
        print("Column 'IMDb Score' not found in DataFrame")

    if not result.empty:
        result = result[['Title', 'Genre', 'Tags', 'Actors', 'ViewerRating', 'Image', 'Netflix Link', 'Summary', 'IMDb Score', 'Director', 'Writer', 'Release Date', 'Runtime']]  # Ensure these columns exist

    
        images = result['Image'].tolist()
        titles = result['Title'].tolist()
        return render_template('result.html', titles=titles, images=images)
    else:
        print("No results found.")
        return render_template('result.html', titles=[], images=[])

@app.route('/moviepage/<name>')
def movie_details(name):
    global result
    # Extracting the details for the show
    if not result.empty:
        details_list = result[result['Title'] == name].iloc[0].tolist() 
        print("Movie Details:", details_list)  
        return render_template('moviepage.html', details=details_list)
    else:
        print("Result is empty")  
        return render_template('moviepage.html', details=[])

if __name__ == '__main__':
    app.run(debug=True)
