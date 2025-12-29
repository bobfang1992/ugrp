"""
Utility to add IMDb and TMDB links to movies.
ML-1M doesn't have links, but ML-20M does. We'll try to match by movieId.
"""

import pandas as pd
from pathlib import Path


def load_links():
    """Load movie links from ML-20M dataset"""
    links_path = Path("data/raw/ml-20m/links.csv")

    if links_path.exists():
        links = pd.read_csv(links_path)
        return links
    else:
        return None


def get_imdb_url(imdb_id):
    """Construct IMDb URL from ID"""
    if pd.isna(imdb_id):
        return None
    # IMDb IDs are stored without leading zeros in links.csv
    # IMDb URLs need 7 digits with leading zeros
    imdb_id_str = str(int(imdb_id)).zfill(7)
    return f"https://www.imdb.com/title/tt{imdb_id_str}/"


def get_tmdb_url(tmdb_id):
    """Construct TMDB URL from ID"""
    if pd.isna(tmdb_id):
        return None
    return f"https://www.themoviedb.org/movie/{int(tmdb_id)}"


def get_imdb_search_url(title, year=None):
    """Construct IMDb search URL from title and year"""
    import urllib.parse
    query = title
    if year and not pd.isna(year):
        query = f"{title} {int(year)}"
    encoded = urllib.parse.quote(query)
    return f"https://www.imdb.com/find?q={encoded}"


def add_links_to_movies(movies_df):
    """Add IMDb and TMDB links to movies DataFrame"""
    links = load_links()

    if links is not None:
        # Merge with links
        movies_with_links = movies_df.merge(
            links[['movieId', 'imdbId', 'tmdbId']],
            on='movieId',
            how='left'
        )

        # Create URL columns
        movies_with_links['imdb_url'] = movies_with_links['imdbId'].apply(get_imdb_url)
        movies_with_links['tmdb_url'] = movies_with_links['tmdbId'].apply(get_tmdb_url)
    else:
        # No links available, create search URLs
        movies_with_links = movies_df.copy()
        movies_with_links['imdbId'] = None
        movies_with_links['tmdbId'] = None
        movies_with_links['imdb_url'] = None
        movies_with_links['tmdb_url'] = None

    # Always add search URL as fallback
    movies_with_links['imdb_search_url'] = movies_with_links.apply(
        lambda row: get_imdb_search_url(row['title_clean'], row.get('year')),
        axis=1
    )

    return movies_with_links
