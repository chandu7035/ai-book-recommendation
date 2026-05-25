import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(

    "books.csv",

    on_bad_lines="skip",

    engine="python"
)

# ==========================================
# USE ONLY 3000 ROWS
# ==========================================

df = df.head(1000)

# ==========================================
# KEEP IMPORTANT COLUMNS
# ==========================================

df = df[[

    "title",

    "authors",

    "average_rating",

    "ratings_count",

    "publisher"
]]

# ==========================================
# REMOVE NULL VALUES
# ==========================================

df.dropna(inplace=True)

# ==========================================
# CONVERT TO STRING
# ==========================================

df["title"] = df[
    "title"
].astype(str)

df["authors"] = df[
    "authors"
].astype(str)

df["publisher"] = df[
    "publisher"
].astype(str)

# ==========================================
# CREATE TAGS
# ==========================================

df["tags"] = (

    df["title"]

    + " "

    + df["authors"]

    + " "

    + df["publisher"]
)

# ==========================================
# LOWERCASE TAGS
# ==========================================

df["tags"] = df[
    "tags"
].str.lower()

# ==========================================
# TEXT VECTORIZATION
# ==========================================

cv = CountVectorizer(

    max_features=5000,

    stop_words="english"
)

vectors = cv.fit_transform(

    df["tags"]

).toarray()

# ==========================================
# COSINE SIMILARITY
# ==========================================

similarity = cosine_similarity(
    vectors
)

# ==========================================
# SAVE FILES
# ==========================================

pickle.dump(

    df,

    open(
        "books.pkl",
        "wb"
    )
)

pickle.dump(

    similarity,

    open(
        "similarity.pkl",
        "wb"
    )
)

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print(
    "Book recommendation model trained successfully"
)
