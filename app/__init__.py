from dotenv import load_dotenv

# Carrega o .env sem sobrescrever o ambiente: variável passada ao container sempre vence
load_dotenv(override=False)
