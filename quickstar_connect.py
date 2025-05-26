
import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()

ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

def init_astra_db():
    if not ASTRA_DB_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        raise RuntimeError("Chybí ASTRA_DB_ENDPOINT nebo ASTRA_DB_APPLICATION_TOKEN! Zkontroluj .env soubor.")
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_DB_ENDPOINT)
    return db
