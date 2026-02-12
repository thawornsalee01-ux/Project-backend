import os
from dotenv import load_dotenv
load_dotenv()

print("AI KEY:", os.getenv("LOCALMODEL_API_KEY"))
