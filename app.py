from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

import pickle
import pandas as pd
import os

# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# TRAIN MODEL IF NOT EXISTS
# ==========================================

if not os.path.exists(
    "books.pkl"
):

    import train

# ==========================================
# LOAD FILES
# ==========================================

books = pickle.load(
    open("books.pkl", "rb")
)

similarity = pickle.load(
    open("similarity.pkl", "rb")
)

# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================================
# RECOMMENDATION ROUTE
# ==========================================

@app.route(
    "/recommend",
    methods=["POST"]
)
def recommend():

    try:

        data = request.json

        book_name = data[
            "book"
        ].lower()

        # ==========================================
        # SEARCH BOOK
        # ==========================================

        matched_books = books[
            books["title"]
            .str.lower()
            .str.contains(book_name)
        ]

        # ==========================================
        # BOOK NOT FOUND
        # ==========================================

        if matched_books.empty:

            return jsonify({

                "error":
                "Book not found"
            })

        # ==========================================
        # GET FIRST MATCHED BOOK
        # ==========================================

        book = matched_books.iloc[
            0
        ]["title"]

        # ==========================================
        # GET INDEX
        # ==========================================

        index = books[
            books["title"]
            == book
        ].index[0]

        # ==========================================
        # GET SIMILARITY SCORES
        # ==========================================

        distances = similarity[
            index
        ]

        # ==========================================
        # GET TOP RECOMMENDATIONS
        # ==========================================

        book_list = sorted(

            list(
                enumerate(distances)
            ),

            reverse=True,

            key=lambda x: x[1]

        )[1:6]

        recommendations = []

        # ==========================================
        # STORE RECOMMENDATIONS
        # ==========================================

        for i in book_list:

            recommendations.append({

                "title":
                books.iloc[i[0]][
                    "title"
                ],

                "author":
                books.iloc[i[0]][
                    "authors"
                ],

                "rating":
                books.iloc[i[0]][
                    "average_rating"
                ]
            })

        # ==========================================
        # RETURN RESULT
        # ==========================================

        return jsonify({

            "recommendations":
            recommendations
        })

    except Exception as e:

        return jsonify({

            "error":
            str(e)
        })

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )
