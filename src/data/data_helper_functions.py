from collections import defaultdict
import pandas as pd
import numpy as np
from ast import literal_eval
import networkx as nx
import os
import pickle
import ast
import math
import re
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """
    Loads the preprocessed books, users, and ratings dataframes from the specified data_path.
    """
    books_df = pd.read_csv(
        data_path + 'processed_books.csv', index_col='book_id')
    users_df = pd.read_csv(
        data_path + 'processed_users.csv', index_col='user_id')
    ratings_df = pd.read_csv(data_path + 'processed_ratings.csv')

    return books_df, users_df, ratings_df

def split_ratings_by_user(ratings_df, test_size=0.2, random_state=0):
    """
    This function splits the given ratings DataFrame into training and test sets for each user, 
    ensuring that a specific proportion of each user's ratings are in the training and test sets. 
    """
    np.random.seed(random_state)
    
    train_ratings_list = []
    test_ratings_list = []
    
    users = ratings_df['user_id'].unique()
    for user in users:
        user_ratings = ratings_df[ratings_df['user_id'] == user]
        train_ratings, test_ratings = train_test_split(user_ratings, test_size=test_size, random_state=random_state)
        
        train_ratings_list.append(train_ratings)
        test_ratings_list.append(test_ratings)
        
    train_ratings_df = pd.concat(train_ratings_list)
    test_ratings_df = pd.concat(test_ratings_list)

    return train_ratings_df, test_ratings_df


def load_bipartite_graph(books_df=None, users_df=None, ratings_df=None, graph_file='../Data/bipartite_graph.pickle'):
    """
    Constructs a bipartite graph representing users and books, and their ratings as weighted edges.
    Tries to load the graph from a file if it exists, otherwise creates and saves the graph.
    """
    if os.path.exists(graph_file):
        G = pickle.load(open(graph_file, 'rb'))
    else:
        G = nx.Graph()
        
        # Add nodes
        for user_id in users_df.index:
            G.add_node(f'u-{user_id}', type='user')  # Users
        for book_id in books_df.index:
            G.add_node(f'b-{book_id}', type='book')  # Books

        # Add edges
        for _, row in ratings_df.iterrows():
            G.add_edge(f'u-{row["user_id"]}', f'b-{row["book_id"]}', weight=row['rating'])

        pickle.dump(G, open(graph_file, 'wb'))
    return G

def load_new_bipartite_graph(books_df=None, users_df=None, ratings_df=None):
    G = nx.Graph()
    
    for user_id in users_df.index:
        G.add_node(f'u-{user_id}', type=0)  # Users
    for book_id in books_df.index:
        G.add_node(f'b-{book_id}', type=1)  # Books

    # Add edges
    for _, row in ratings_df.iterrows():
        G.add_edge(f'u-{row["user_id"]}', f'b-{row["book_id"]}', weight=row['rating'])

    # Apply the sigmoid function to user ratings to emphasize ratings above users average, books users enjoyed more
    for node, data in G.nodes(data=True):
        if data['type'] != 0:
            continue
        user_edges = G.edges(node, data=True)
        total_weight = sum(edge_data['weight'] for _, _, edge_data in user_edges)
        average_rating = total_weight / len(user_edges)
        sigmoid_weights = [sigmoid(edge_data['weight'], average_rating) for _, _, edge_data in user_edges]
        total_sigmoid_weight = sum(sigmoid_weights)
        
        for (_, _, edge_data), sigmoid_weight in zip(user_edges, sigmoid_weights):
            edge_data['sigmoid_weight'] = sigmoid_weight
            edge_data['normalized_weight'] = edge_data['weight'] / total_weight
            edge_data['normalized_sigmoid_weight'] = sigmoid_weight / total_sigmoid_weight

    return G

def preprocess_author_name(author_name):
    """
    Remove additional spaces between names and usernames of some authors
    """
    author_name = author_name.strip()
    return re.sub(r'\s+', ' ', author_name)

def get_authors(books_df):
    """
    Extract all unique authors from the books_df dataframe.
    """
    authors_set = set()
    for authors in books_df['authors']:
        book_authors = [preprocess_author_name(author) for author in authors.split(',')]
        authors_set.update(book_authors)
    return authors_set

def get_authors_of_book(book_id, books_df):
    """
    Extract authors of a specific book 
    """
    return [preprocess_author_name(author) for author in books_df.loc[book_id]['authors'].split(',')]

def get_genres(books_df):
    """
    Extract all unique genres from the books_df dataframe.
    """
    genres_set = set()
    for genres in books_df['genres']:
        genre_list = ast.literal_eval(genres)
        genres_set.update(genre_list)
    return genres_set

def get_genres_of_book(book_id, books_df):
    """
    Extract genres of a specific book 
    """
    return ast.literal_eval(books_df.loc[book_id]['genres'])
    

def get_year_intervals():
    """
    Get year intervals.
    """
    return ['<1900', '1900-1949', '1950-1999', '2000-2009', '2010+']

def get_book_length_intervals():
    """
    Get book length intervals based on the number of pages.
    """
    return ['<149', '150-499', '500+']

def sigmoid(rating, average_rating):
    weight_difference = rating - average_rating
    sigmoid_factor = 1 / (1 + math.exp(-weight_difference))
    return rating * sigmoid_factor

def load_multipartite_graph(books_df=None, users_df=None, ratings_df=None, graph_file='../Data/multipartite_graph.pickle', create=False):
    """
    Loads a multipartite graph with nodes representing users, books, authors, and genres.
    The edges are created based on user ratings, book-author relationships, and book-genre relationships.
    Users' ratings are transformed using the sigmoid function to emphasize ratings above users' average.
    """
    if os.path.exists(graph_file) and not create:
        G = pickle.load(open(graph_file, 'rb'))
    else:
        G = nx.Graph()
        
        # Add nodes
        for user_id in users_df.index:
            G.add_node(f'u-{user_id}', type='user')  # Users
        for book_id in books_df.index:
            G.add_node(f'b-{book_id}', type='book')  # Books
        for author in get_authors(books_df):
            G.add_node(f'a-{author}', type='author') # Authors
        for genre in get_genres(books_df):
            G.add_node(f'g-{genre}', type='genre') # Genres
        # for language in books_df['language'].unique():
        #     G.add_node(f'l-{language}', type='language') # Language
        # for year_interval in get_year_intervals():
        #     G.add_node(f'y-{year_interval}', type='year_interval') # Year intervals
        # for pages_interval in get_book_length_intervals():
        #     G.add_node(f'p-{pages_interval}', type='pages_interval') # Pages intervals
        
        # Add user-book edges
        for _, row in ratings_df.iterrows():
            G.add_edge(f'u-{row["user_id"]}', f'b-{row["book_id"]}', weight=row['rating'])

        # Apply the sigmoid function to user ratings to emphasize ratings above users average, books users enjoyed more
        for node, data in G.nodes(data=True):
            if data['type'] != 'user':
                continue

            user_edges = G.edges(node, data=True)
            total_weight = sum(edge_data['weight'] for _, _, edge_data in user_edges)
            average_rating = total_weight / len(user_edges)
            sigmoid_weights = [sigmoid(edge_data['weight'], average_rating) for _, _, edge_data in user_edges]

            for (_, _, edge_data), sigmoid_weight in zip(user_edges, sigmoid_weights):
                edge_data['ppr_weight'] = sigmoid_weight

        # Add book-author edges
        book_author_weight = 0.5
        for book_id, row in books_df.iterrows():
            authors = get_authors_of_book(book_id, books_df)
            for author in authors:
                G.add_edge(f'b-{book_id}', f'a-{author}', ppr_weight=book_author_weight)
        
        # Add book-author edges
        book_genre_weight = 0.5
        for book_id, row in books_df.iterrows():
            genres = get_genres_of_book(book_id, books_df)
            for genre in genres:
                G.add_edge(f'b-{book_id}', f'g-{genre}', ppr_weight=book_genre_weight)

        pickle.dump(G, open(graph_file, 'wb'))
    return G


def load_data_original(data_path, save=False, filter=False):
    """
    Loads and preprocesses dataframes: books and ratings from the specified data_path.
    Preprocessing steps include:
    1. Selecting relevant columns and renaming them.
    2. Extracting additional book information and adding new columns.
    3. Filling in missing values for the books dataframe.
    4. Filtering users with a minimum number of ratings. 
    5. Creating a users dataframe containing user statistics.
    """

    books_df = pd.read_csv(data_path + 'books.csv', low_memory=False)
    ratings_df = pd.read_csv(data_path + 'ratings.csv',
                             low_memory=False, index_col=False)

    # Select only relavant columns and rename them
    books_df = books_df[['id', 'authors', 'original_publication_year', 'original_title', 'title',
                         'average_rating', 'ratings_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']]
    books_df = books_df.rename(columns={
        'id': 'book_id',
        'original_publication_year': 'year',
        'title': 'secondary_title',
        'original_title': 'title',
        'ratings_1': 'num_1',
        'ratings_2': 'num_2',
        'ratings_3': 'num_3',
        'ratings_4': 'num_4',
        'ratings_5': 'num_5',
        'ratings_count': 'num_ratings',
    })
    books_df.set_index('book_id', inplace=True)

    # Extract additional book information and add new columns
    additional_book_info_df = extract_additional_info()
    books_df['genres'] = additional_book_info_df['genres']
    books_df['language'] = additional_book_info_df['language_code']
    books_df['pages'] = additional_book_info_df['pages']

    # Fill in missing values
    average_year = books_df['year'].mean()
    # Gaussian distribution
    noise_years = np.random.uniform(-5, 5, books_df['year'].isnull().sum())
    books_df.loc[books_df['year'].isnull(), 'year'] = (
        average_year + noise_years).round().astype(int)

    books_df['language'].fillna('eng', inplace=True)

    # Use secondary title is title is missing
    books_df['title'].fillna(books_df['secondary_title'], inplace=True)
    books_df.drop('secondary_title', axis=1, inplace=True)

    average_pages = books_df['pages'].mean()
    # Gaussian distribution
    noise_pages = np.random.uniform(-50, 50, books_df['pages'].isnull().sum())
    books_df.loc[books_df['pages'].isnull(), 'pages'] = (
        average_pages + noise_pages).round().astype(int)

    # Only one book has missing genre
    books_df.at[10000, 'genres'] = ['world war', '1914-1918']

    if filter:  # Filter users
        user_rating_counts = ratings_df['user_id'].value_counts()
        users_with_enough_ratings = user_rating_counts[user_rating_counts >= 4].index
        ratings_df = ratings_df[ratings_df['user_id'].isin(
            users_with_enough_ratings)]

    # Create users_df with unique user IDs from ratings_df and calculate user statistics
    users_df = pd.DataFrame(
        ratings_df['user_id'].unique(), columns=['user_id'])
    users_df.set_index('user_id', inplace=True)
    user_stats = ratings_df.groupby('user_id').agg(
        {'rating': ['count', 'mean', 'std', 'median']})
    user_stats.columns = user_stats.columns.droplevel(0)
    users_df = users_df.merge(user_stats, left_index=True, right_index=True)
    users_df = users_df.rename(columns={
        'count': 'num_ratings',
        'mean': 'average_rating',
        'median': 'median_rating'
    })

    if save:
        books_df.to_csv(data_path + 'processed_books.csv')
        users_df.to_csv(data_path + 'processed_users.csv')
        ratings_df.to_csv(data_path + 'processed_ratings.csv', index=False)

    return books_df, users_df, ratings_df


def extract_additional_info():
    return pd.read_csv('https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv', index_col=[0], converters={"genres": literal_eval})
