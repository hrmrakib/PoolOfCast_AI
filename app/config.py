import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "DATABASE_LOCAL_URL")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
BASE_URL = os.getenv("BASE_URL")
BASE_URL_BACKEND = os.getenv("BASE_URL_BACKEND")
OPENAI_CHAT_MODEL = "gpt-5.1"  #gpt-5-mini
# JWT Authentication
JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"

# Email Configuration for Notifications
SMTP_SERVER = os.getenv("EMAIL_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_PORT", 587))
SMTP_USERNAME = os.getenv("EMAIL_HOST_USER")
SMTP_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")
SENDER_EMAIL = os.getenv("DEFAULT_FROM_EMAIL")
