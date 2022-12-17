import pycountry
import pandas as pd
import pycountry
import pandas as pd
import os.path
import re

def find_iso(x):
    """
    find matching country iso codes

    x: a country name (str)
    """
    try:
        country = pycountry.countries.search_fuzzy(x)
        return country[0].alpha_3
    except:
        return None

def pickle_load(path, load_with_text=False):
    """
    loads the corresponding pickle file or creates it if it does not exist
    
    
    path: path to original txt file   

    returns: dataframe with the files content
    """
    pickle_path = path + ".pickle"
    if load_with_text and ("rating" or "review" in path):
        pickle_path += "s"
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)
    else:
        if "ratings" or "review" in path:
            # works for reveiws and ratings, drops the text

            c = open(path, "r").read()
            
            re_data = re.findall(r"^beer_name.+?(?=\n\n)", c,  re.DOTALL | re.MULTILINE) # includes everything...
            
            def get_attributes(data):
                return dict(re.findall(r"^([a-zA-Z][^:]+):\s*(.*?)\s*?(?=^[a-zA-Z][^:]+:|\Z)", data,  re.DOTALL | re.MULTILINE))

            elements = [get_attributes(data) for data in re_data]
            
            df = pd.DataFrame(elements)

            df["date"] = pd.to_datetime(df["date"],unit='s')
            for col in ["abv","appearance","aroma","palate","taste","overall","rating"]:
                df[col] = pd.to_numeric(df[col], errors = 'coerce')
            df.set_index(df["date"], inplace = True)
            df.to_pickle(pickle_path)
            return df
        else:
            df = pd.read_csv(path, sep=",", low_memory=False)
            df.to_pickle(pickle_path)
            return df


def isUTF8(data):
    try:
        data.decode('UTF-8')
    except UnicodeDecodeError:
        return False
    else:
        return True

from codecs import BOM_UTF8, BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE

BOMS = (
    (BOM_UTF8, "UTF-8"),
    (BOM_UTF32_BE, "UTF-32-BE"),
    (BOM_UTF32_LE, "UTF-32-LE"),
    (BOM_UTF16_BE, "UTF-16-BE"),
    (BOM_UTF16_LE, "UTF-16-LE"),
)

def isASCII(data):
    try:
        data.decode('ASCII')
    except UnicodeDecodeError:
        return False
    else:
        return True

def check_bom(data):
    return [encoding for bom, encoding in BOMS if data.startswith(bom)]

def determine_encoding(filename):
    with open(filename, "rb") as f: # Reads in binary format
        data = f.read()
        if isUTF8(data):
            print("UTF-8")
        elif isASCII(data):
            print("ASCII")
        else:
            print("UTF-16")
        print(check_bom(data))



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
    filename = path_ba + "ratings.txt" # -> is UTF-8 / no BOM
    # filename = path_rb + "ratings.txt" # -> is UTF-8 / no BOM
    # filename = path_rb + "reviews.txt" # -> is UTF-8 / no BOM
    # df = pickle_load(filename, load_with_text=True)  

    # print(df.head())
    # print(df.describe())
    # print(list(df.columns))
    determine_encoding(filename)

