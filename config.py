from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ETF_FILES = {
    "SPY": RAW_DATA_DIR / "SPY.csv",
    "QQQ": RAW_DATA_DIR / "QQQ.csv",
    "DIA": RAW_DATA_DIR / "DIA.csv",
    "IWN": RAW_DATA_DIR / "IWN.csv",
}