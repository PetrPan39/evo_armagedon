
import os
from dotenv import load_dotenv

# Načtení proměnných z .env
load_dotenv()

ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_REGION = os.getenv("ASTRA_DB_REGION")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def connect_to_astra():
    cloud_config = {
        'secure_connect_bundle': f'./secure-connect-{ASTRA_DB_ID}/'
    }
    auth_provider = PlainTextAuthProvider('token', ASTRA_DB_APPLICATION_TOKEN)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    session.set_keyspace(ASTRA_DB_KEYSPACE)
    return session

# Inicializace databáze
try:
    session = connect_to_astra()
    print("✅ Připojeno k Astra DB")
except Exception as e:
    print("❌ Chyba při připojení k Astra DB:", e)

# Inicializace dalších modulů (dle potřeby)
try:
    import api_server
    api_server.run()
except ImportError:
    print("⚠️ Modul api_server nelze spustit – ujistěte se, že existuje a má funkci run().")
