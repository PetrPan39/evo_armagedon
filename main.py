
import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

# Načtení proměnných z .env
load_dotenv()

ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

def connect_to_astra():
    if not ASTRA_DB_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        raise RuntimeError("Chybí ASTRA_DB_ENDPOINT nebo ASTRA_DB_APPLICATION_TOKEN! Zkontroluj .env.")
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_DB_ENDPOINT)
    print(f"✅ Připojeno k Astra DB: {db.list_collection_names()}")
    return db

try:
    db = connect_to_astra()
except Exception as e:
    print("❌ Chyba při připojení k Astra DB:", e)
    db = None

# Spuštění API serveru nebo hlavního systému
if __name__ == "__main__":
    import api_server
    api_server.run()
