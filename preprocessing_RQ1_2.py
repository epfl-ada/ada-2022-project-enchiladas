# %% [markdown]
#  # Rescale Ratings
# Some users rate on average more positively or more negativly with respect to others. Some of this assumed effect will occur, because users rate different beers which indeed are better or worse on average. We hereby assume, that some users are just more positive or negative in general, even if they rate the same beers.
# To counteract this effect, we rescale the ratings to a -1,1 scale where 0 is the users average rating. This way, we can compare users on an equal footing.

# TODO: Formalize what we do. Ideally add citations.
# %%
def scale(rating, user_average, top=5, bottom=1):
    """
    Returns the scale of the rating with respect to the users average rating.
    """
    if rating > user_average:
        return top-user_average
    elif rating < user_average:
        return user_average-bottom
    else:
        return bottom
def rescale(rating, user_average, top = 5, bottom = 1):
    """
    Rescales the rating to a -1,1 scale where 0 is the users average rating.
    """
    return (rating - user_average) / scale(rating, user_average, top, bottom)

# %% [markdown]
#  ## BA
# Columns to be rescaled in BA: 
# - aroma (1-5)
# - appearance (1-5)
# - taste (1-5)
# - palate (1-5)
# - overall (1-5)
# - rating (1-5)

# TODO: here might be a good place to explain the different usages of the individual ratings/aspects across the two datasets

# %%
# Sanity check the ranges of the ratings and aspects
print("aroma", df_ba["aroma"].min(), df_ba["aroma"].max())
print("appearance", df_ba["appearance"].min(), df_ba["appearance"].max())
print("taste", df_ba["taste"].min(), df_ba["taste"].max())
print("palate", df_ba["palate"].min(), df_ba["palate"].max())
print("overall", df_ba["overall"].min(), df_ba["overall"].max())
print("rating", df_ba["rating"].min(), df_ba["rating"].max())

# %%
# Compute the average rating for each user
df_ba["user_average_aroma"] = df_ba.groupby("user_id")["aroma"].transform("mean")
df_ba["user_average_appearance"] = df_ba.groupby("user_id")["appearance"].transform("mean")
df_ba["user_average_taste"] = df_ba.groupby("user_id")["taste"].transform("mean")
df_ba["user_average_palate"] = df_ba.groupby("user_id")["palate"].transform("mean")
df_ba["user_average_overall"] = df_ba.groupby("user_id")["overall"].transform("mean")
df_ba["user_average_rating"] = df_ba.groupby("user_id")["rating"].transform("mean")

# %%
# Rescale the ratings and aspects to a -1,1 scale where 0 is the users average rating

df_ba["aroma_rescaled"] = df_ba.apply(lambda row: rescale(row["aroma"], row["user_average_aroma"]), axis=1)
df_ba["appearance_rescaled"] = df_ba.apply(lambda row: rescale(row["appearance"], row["user_average_appearance"]), axis=1)
df_ba["taste_rescaled"] = df_ba.apply(lambda row: rescale(row["taste"], row["user_average_taste"]), axis=1)
df_ba["palate_rescaled"] = df_ba.apply(lambda row: rescale(row["palate"], row["user_average_palate"]), axis=1)
df_ba["overall_rescaled"] = df_ba.apply(lambda row: rescale(row["overall"], row["user_average_overall"]), axis=1)
df_ba["rating_rescaled"] = df_ba.apply(lambda row: rescale(row["rating"], row["user_average_rating"]), axis=1)

# %%
# Sanity check the ranges of the rescaled ratings and aspects
print("aroma_rescaled", df_ba["aroma_rescaled"].min(), df_ba["aroma_rescaled"].max())
print("appearance_rescaled", df_ba["appearance_rescaled"].min(), df_ba["appearance_rescaled"].max())
print("taste_rescaled", df_ba["taste_rescaled"].min(), df_ba["taste_rescaled"].max())
print("palate_rescaled", df_ba["palate_rescaled"].min(), df_ba["palate_rescaled"].max())
print("overall_rescaled", df_ba["overall_rescaled"].min(), df_ba["overall_rescaled"].max())
print("rating_rescaled", df_ba["rating_rescaled"].min(), df_ba["rating_rescaled"].max())

# %% [markdown]
#  ## RB
# Columns to be rescaled in RB:
# - aroma (1-10)
# - appearance (1-5)
# - taste (1-10)
# - palate (1-5)
# - overall (1-20)
# - rating (0-5)

# %%
# Add 1 to every rating to make it 1-6 instead of 0-5 to prevent divison by zero
df_rb["translated_rating"] = df_rb["rating"] + 1

# %%
# Sanity check the ranges of the ratings and aspects
print("aroma", df_rb["aroma"].min(), df_rb["aroma"].max())
print("appearance", df_rb["appearance"].min(), df_rb["appearance"].max())
print("taste", df_rb["taste"].min(), df_rb["taste"].max())
print("palate", df_rb["palate"].min(), df_rb["palate"].max())
print("overall", df_rb["overall"].min(), df_rb["overall"].max())
print("translated_rating", df_rb["translated_rating"].min(), df_rb["translated_rating"].max())

# %%
# Compute the average rating for each user
df_rb["user_average_aroma"] = df_rb.groupby("user_id")["aroma"].transform("mean")
df_rb["user_average_appearance"] = df_rb.groupby("user_id")["appearance"].transform("mean")
df_rb["user_average_taste"] = df_rb.groupby("user_id")["taste"].transform("mean")
df_rb["user_average_palate"] = df_rb.groupby("user_id")["palate"].transform("mean")
df_rb["user_average_overall"] = df_rb.groupby("user_id")["overall"].transform("mean")
df_rb["user_average_rating"] = df_rb.groupby("user_id")["translated_rating"].transform("mean")

# %%
# Rescale the ratings and aspects to a -1,1 scale where 0 is the users average rating

df_rb["aroma_rescaled"] = df_rb.apply(lambda row: rescale(row["aroma"], row["user_average_aroma"],top=10), axis=1)
df_rb["appearance_rescaled"] = df_rb.apply(lambda row: rescale(row["appearance"], row["user_average_appearance"]), axis=1)
df_rb["taste_rescaled"] = df_rb.apply(lambda row: rescale(row["taste"], row["user_average_taste"],top=10), axis=1)
df_rb["palate_rescaled"] = df_rb.apply(lambda row: rescale(row["palate"], row["user_average_palate"]), axis=1)
df_rb["overall_rescaled"] = df_rb.apply(lambda row: rescale(row["overall"], row["user_average_overall"],20), axis=1)
df_rb["rating_rescaled"] = df_rb.apply(lambda row: rescale(row["translated_rating"], row["user_average_rating"],top=6), axis=1)

# %%
# Sanity check the ranges of the rescaled ratings and aspects
print("aroma_rescaled", df_rb["aroma_rescaled"].min(), df_rb["aroma_rescaled"].max())
print("appearance_rescaled", df_rb["appearance_rescaled"].min(), df_rb["appearance_rescaled"].max())
print("taste_rescaled", df_rb["taste_rescaled"].min(), df_rb["taste_rescaled"].max())
print("palate_rescaled", df_rb["palate_rescaled"].min(), df_rb["palate_rescaled"].max())
print("overall_rescaled", df_rb["overall_rescaled"].min(), df_rb["overall_rescaled"].max())
print("rating_rescaled", df_rb["rating_rescaled"].min(), df_rb["rating_rescaled"].max())

# %% [markdown]
# It is normal that the min of the rescaled ratings is -0.82, the worst score in the rb df is not 0, but 0.5. I.e. nobody rated a beer with 0 stars.
