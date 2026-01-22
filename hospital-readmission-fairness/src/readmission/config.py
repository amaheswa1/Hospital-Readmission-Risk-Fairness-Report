from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    raw_data_path: str = os.getenv("RAW_DATA_PATH", "data/raw/readmissions.csv")
    processed_data_dir: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")
    report_dir: str = os.getenv("REPORT_DIR", "reports")

settings = Settings()
