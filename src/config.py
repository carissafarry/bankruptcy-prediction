# config.py
import os
import json

PREFIX = "APP_"


def _cast(value, cast):
    if cast is list:
        try:
            return json.loads(value)
        except Exception:
            return [v.strip() for v in value.split(",") if v.strip()]
    return cast(value)

def load_env(prefix=PREFIX):
    """
    Load all APP_ envs.
    : APP_GOOGLE_CREDS_PATH -> GOOGLE_CREDS_PATH
    """
    return {
        key[len(prefix):]: value
        for key, value in os.environ.items()
        if key.startswith(prefix)
    }


_RAW_ENV = load_env()

def get(key, default=None, cast=None):
    value = _RAW_ENV.get(key, default)

    if value is None:
        return default

    if cast:
        try:
            return _cast(value, cast)
        except Exception as e:
            raise RuntimeError(
                f"Invalid value for env '{key}': {value}"
            ) from e

    return value


# NEGATIVE KEYWORDS
ENV_NEGATIVE_KEYWORDS = get("NEGATIVE_KEYWORDS", None, list)
DEFAULT_NEGATIVE_KEYWORDS = [
    "gagal bayar", "kredit macet", "non performing loan", "npl",
    "likuiditas", "kerugian", "rugi", "penurunan laba",
    "bangkrut", "pailit",
    "fraud", "penipuan", "korupsi", "skandal",
    "pidana", "tersangka", "ditahan", "penyidikan",
    "denda", "sanksi", "dibekukan", "pencabutan izin",
    "penutupan", "tutup", "dihentikan",
    "krisis", "guncangan", "gagal", "masalah",
]
NEGATIVE_KEYWORDS_FINAL = (
    DEFAULT_NEGATIVE_KEYWORDS
    if not ENV_NEGATIVE_KEYWORDS
    else list(set(DEFAULT_NEGATIVE_KEYWORDS + ENV_NEGATIVE_KEYWORDS))
)


# CONFIG
CONFIG = {
    # Google
    "GOOGLE_CREDS_PATH": get("GOOGLE_CREDS_PATH", "service_account.json"),
    "SPREADSHEET_NAME": get("SPREADSHEET_NAME", "bank_news_scrapping_data"),
    "SHEET_NAME": get("SHEET_NAME", "Sheet1"),

    # Columns
    "COL_FIRST_SEEN": get("COL_FIRST_SEEN", 1, int),
    "COL_LAST_SEEN": get("COL_LAST_SEEN", 2, int),
    "COL_PUBLISHED_AT": get("COL_PUBLISHED_AT", 3, int),
    "COL_SOURCE": get("COL_SOURCE", 4, int),
    "COL_YEAR": get("COL_YEAR", 5, int),
    "COL_QUARTER": get("COL_QUARTER", 6, int),
    "COL_SYMBOL": get("COL_SYMBOL", 7, int),
    "COL_TITLE": get("COL_TITLE", 8, int),
    "COL_IS_NEGATIVE": get("COL_IS_NEGATIVE", 9, int),
    "COL_NEG_KEYWORD": get("COL_NEG_KEYWORD", 10, int),
    "COL_LINK": get("COL_LINK", 11, int),

    # Scraping
    "SCRAPING_LIMIT": get("SCRAPING_LIMIT", 100, int),
    "NEGATIVE_KEYWORDS": NEGATIVE_KEYWORDS_FINAL,

    # Runtime
    "TIMEOUT_REQUEST": get("TIMEOUT_REQUEST", 15, int),
    "TIMEZONE": get("TIMEZONE", "Asia/Jakarta"),
}

