import json
from psycopg2 import connect, InternalError
from contextlib import contextmanager
import pandas as pd
from dotenv import load_dotenv
import os
from helper.utils import *


# Load environment variables from .env file
load_dotenv()

connection_dict = {
    "database": os.getenv("REDSHIFT_DATABASE"),
    "host": os.getenv("REDSHIFT_HOST"),
    "password": os.getenv("REDSHIFT_PASSWORD"),
    "port": int(os.getenv("REDSHIFT_PORT")),  # Ensure the port is an integer
    "user": os.getenv("REDSHIFT_USER")
}


# Define a context manager for connecting to Redshift
@contextmanager
def redshift_connect_scope(readonly: bool = False, **kwargs):
    """
    Connects to Redshift and returns the connection object as a context manager.
    :param readonly: Sets the connection to readonly mode if True.
    :param kwargs: Additional arguments for the connection.
    :return: Connection object.
    """

    conn = connect(**connection_dict)

    if readonly:
        conn.set_session(readonly=True)

    try:
        yield conn
        conn.commit()
    except InternalError as e:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

# Using the context manager
with redshift_connect_scope(readonly=True) as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT * "
                       "FROM steam_review "
                       "where app_id_name = '1928980_Nightingale' LIMIT 5000")  # Run the query

        reviews = cursor.fetchall()  # Fetch all the results
        columns = [desc[0] for desc in cursor.description]  # Get column names

# Convert the result to a JSON string
reviews_json = json.dumps([dict(zip(columns, row)) for row in reviews], indent=4)
print("JSON Output:")
print(reviews_json)

# Convert the result to a Pandas DataFrame
df = pd.DataFrame(reviews, columns=columns)
print("DataFrame Output:")
print(df)

#Define path
root_dir = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis\Data\Steamapps'
steam_title = 'Market'

path_db_prepared = os.path.join(root_dir, steam_title, "db_prepared.json")

save_df_as_json(df, path_db_prepared)