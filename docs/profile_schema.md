# User Profile JSON Schema

User profiles are stored in `data/processed/user_profiles.json` as a dictionary mapping `userId` (int) to profile objects.

## Schema Definition

```json
{
  "<userId>": {
    "userId": int,
    "num_ratings": int,
    "avg_rating": float,
    "std_rating": float,
    "top_genres": {
      "<genre>": float,
      ...
    },
    "year_min": int | null,
    "year_max": int | null,
    "year_median": float | null,
    "top_decades": {
      "<decade>": int,
      ...
    },
    "popularity_bias": float,
    "exploration_score": float,
    "exploration_tendency": "conservative" | "medium" | "adventurous",
    "recent_liked": [
      {
        "movieId": int,
        "title": string,
        "year": int | null,
        "rating": int,
        "genres": [string, ...]
      },
      ...
    ]
  }
}
```

## Field Descriptions

### Core Stats

| Field | Type | Description |
|-------|------|-------------|
| `userId` | `int` | User ID (matches ML-1M userId) |
| `num_ratings` | `int` | Total number of ratings by this user |
| `avg_rating` | `float` | Mean rating (1-5 scale) |
| `std_rating` | `float` | Standard deviation of ratings |

### Genre Preferences

| Field | Type | Description |
|-------|------|-------------|
| `top_genres` | `dict[str, float]` | Top 5 genres with proportions (0-1) from liked items (rating ≥ 4). Key=genre name, Value=proportion of liked items containing this genre. Sum may exceed 1.0 since movies can have multiple genres. |

**Example:**
```json
"top_genres": {
  "Drama": 0.41,
  "Action": 0.23,
  "Comedy": 0.18,
  "Thriller": 0.12,
  "Adventure": 0.08
}
```

### Temporal Preferences

| Field | Type | Description |
|-------|------|-------------|
| `year_min` | `int \| null` | Earliest year of liked movies. `null` if no liked items or no year data. |
| `year_max` | `int \| null` | Latest year of liked movies. `null` if no liked items or no year data. |
| `year_median` | `float \| null` | Median year of liked movies. `null` if no liked items or no year data. |
| `top_decades` | `dict[int, int]` | Top 3 decades from liked items. Key=decade (e.g., 1990), Value=count of liked movies from that decade. |

**Example:**
```json
"year_min": 1975,
"year_max": 2000,
"year_median": 1992.0,
"top_decades": {
  "1990": 28,
  "1980": 15,
  "1970": 7
}
```

### Behavioral Metrics

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `popularity_bias` | `float` | `[-1.0, 1.0]` | Preference for popular vs. niche movies. Computed as `(avg_popularity - 0.5) * 2` where `avg_popularity` is the mean popularity quantile of rated items.<br>**-1.0**: Strong preference for niche/indie movies<br>**0.0**: Neutral<br>**+1.0**: Strong preference for popular/blockbuster movies |
| `exploration_score` | `float` | `[0.0, 1.0]` | Composite score measuring adventurousness in taste. Weighted combination of:<br>- 50%: Niche consumption (1 - avg popularity)<br>- 30%: Rating variance<br>- 20%: Genre diversity<br>Requires ≥20 ratings to compute, otherwise 0.0 |
| `exploration_tendency` | `string` | `{conservative, medium, adventurous}` | Categorical version of `exploration_score`:<br>**conservative**: score < 0.33<br>**medium**: 0.33 ≤ score < 0.66<br>**adventurous**: score ≥ 0.66 |

### Recent Activity

| Field | Type | Description |
|-------|------|-------------|
| `recent_liked` | `array[object]` | Up to 10 most recently liked movies (rating ≥ 4), sorted by timestamp descending. Each object contains: |

**Recent liked item schema:**
```json
{
  "movieId": int,        // Movie ID
  "title": string,       // Movie title (without year suffix)
  "year": int | null,    // Release year (null if unavailable)
  "rating": int,         // User's rating (4 or 5)
  "genres": [string]     // List of genres
}
```

## Complete Example

```json
{
  "1": {
    "userId": 1,
    "num_ratings": 53,
    "avg_rating": 4.188679245283019,
    "std_rating": 0.8773609855912358,
    "top_genres": {
      "Drama": 0.20689655172413793,
      "Children's": 0.1724137931034483,
      "Animation": 0.13793103448275862,
      "Comedy": 0.13793103448275862,
      "Adventure": 0.10344827586206896
    },
    "year_min": 1937,
    "year_max": 2000,
    "year_median": 1989.5,
    "top_decades": {
      "1990": 18,
      "1980": 12,
      "1970": 4
    },
    "popularity_bias": 0.7877358490566038,
    "exploration_score": 0.33109756097560976,
    "exploration_tendency": "medium",
    "recent_liked": [
      {
        "movieId": 2396,
        "title": "Shakespeare in Love",
        "year": 1998,
        "rating": 4,
        "genres": ["Comedy", "Romance"]
      },
      {
        "movieId": 1200,
        "title": "Aliens",
        "year": 1986,
        "rating": 5,
        "genres": ["Action", "Horror", "Sci-Fi", "Thriller", "War"]
      }
    ]
  }
}
```

## Usage Notes

1. **Null values**: `year_min`, `year_max`, `year_median` can be `null` if the user has no liked items or if year data is unavailable.

2. **Liked items threshold**: Genre and temporal preferences are computed only from items rated ≥ 4 (out of 5).

3. **Popularity quantile**: Movies are ranked by number of ratings, then normalized to [0, 1] where 0 = least popular, 1 = most popular.

4. **Genre proportions**: Since movies can have multiple genres, the sum of `top_genres` values can exceed 1.0. Each value represents the proportion of liked items that contain that genre.

5. **Exploration score requirements**: Users with fewer than 20 ratings will have `exploration_score = 0.0` and `exploration_tendency = "conservative"` by default (insufficient data).

## Loading Profiles

```python
import json

# Load all profiles
with open("data/processed/user_profiles.json") as f:
    profiles = json.load(f)

# Access specific user
user_profile = profiles["1"]  # Note: keys are strings in JSON
print(f"User 1 prefers: {user_profile['exploration_tendency']} exploration")
```

Or use the ProfileBuilder:

```python
from ugrp.profile.profile_builder import ProfileBuilder

profiles = ProfileBuilder.load_profiles("data/processed/user_profiles.json")
user_profile = profiles[1]  # Integer key after loading
```
