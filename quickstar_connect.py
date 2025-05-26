import os
from dotenv import load_dotenv
from astrapy import DataAPIClient, Database

load_dotenv()

ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
ASTRA_DB_TOKEN    = os.getenv("ASTRA_DB_TOKEN")

if not ASTRA_DB_ENDPOINT or not ASTRA_DB_TOKEN:
    raise RuntimeError(
        "Chybí ASTRA_DB_ENDPOINT nebo ASTRA_DB_TOKEN! "
        "Zkontroluj .env soubor nebo proměnné prostředí."
    )

def connect_to_database() -> Database:
    client = DataAPIClient()
    db = client.get_database(ASTRA_DB_ENDPOINT, token=ASTRA_DB_TOKEN)
    info = db.info()
    print(f"✅ Připojeno k Astra DB: {info.name} ({info.id})")
    return db

def init_astra_db() -> Database:
    return connect_to_database()
