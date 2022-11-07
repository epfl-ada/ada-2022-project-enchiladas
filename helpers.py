import pycountry


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