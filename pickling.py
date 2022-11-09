import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import imageio
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import mapclassify as mc
import os.path
import re

def pickle_load(path):
    """
    loads the corresponding pickle file or creates it if it does not exist
    
    
    path: path to original txt file   

    returns: dataframe with the files content
    """
    pickle_path = path + ".pickles"
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)
    else:
        if "ratings" or "review" in path:
            # works for reveiws and ratings, drops the text

            c = open(filename, "r").read()
            
            # re_data = re.findall(r"^beer_name.+?(?=text)", c,  re.DOTALL | re.MULTILINE)
            re_data = re.findall(r"^beer_name.+?(?=\n\n)", c,  re.DOTALL | re.MULTILINE) # includes everything...
            
            def get_attributes(data):
                return dict(re.findall(r"^([a-zA-Z][^:]+):\s*(.*?)\s*?(?=^[a-zA-Z][^:]+:|\Z)", data,  re.DOTALL | re.MULTILINE))

            elements = [get_attributes(data) for data in re_data]
            
            df = pd.DataFrame(elements)

            df["date"] = pd.to_datetime(df["date"],unit='s')
            for col in ["abv","appearance","aroma","palate","taste","overall","rating"]:
                df[col] = pd.to_numeric(df[col], errors = 'ignore')
            df.set_index(df["date"], inplace = True)
            df.to_pickle(pickle_path)
            return df
        else:
            df = pd.read_csv(path, sep=",", low_memory=False)
            df.to_pickle(pickle_path)
            return df

if __name__ == "__main__":
    data_folder = './Data/'
    path_ba = data_folder + 'BeerAdvocate/'
    path_rb = data_folder + 'RateBeer/'
    path_md = data_folder + 'matched_beer_data/'
    print("running standalone pickler") 
    # filename = path_ba + "ratings.txt"
    # filename = "test.txt"
    # filename = path_md + "ratings_ba.txt"
    filename = path_md + "ratings_rb.txt"
    # filename = path_ba + "reviews.txt"
    # filename = path_ba + "ratings.txt"
    # filename = path_rb + "ratings.txt"
    # filename = path_rb + "reviews.txt"
    df = pickle_load(filename)  

    print(df.head())
    print(df.describe())



