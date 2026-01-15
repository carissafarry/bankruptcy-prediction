import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from scraper.google_news import scrape_google_news

import gspread
from google.oauth2.service_account import Credentials


COL_FIRST_SEEN = 1
COL_LAST_SEEN = 2
COL_PUBLISHED_AT = 3
COL_SOURCE = 4
COL_YEAR = 5
COL_QUARTER = 6
COL_TITLE = 7
COL_IS_NEGATIVE = 8
COL_NEG_REASON = 9
COL_LINK = 10
SCRAPING_LIMIT = 50

NEGATIVE_KEYWORDS = [
    "gagal bayar", "kredit macet", "non performing loan", "npl",
    "likuiditas", "kerugian", "rugi", "penurunan laba",
    "bangkrut", "pailit",
    "fraud", "penipuan", "korupsi", "skandal",
    "pidana", "tersangka", "ditahan", "penyidikan",
    "denda", "sanksi", "dibekukan", "pencabutan izin",
    "penutupan", "tutup", "dihentikan",
    "krisis", "guncangan", "gagal", "masalah",
]


def check_negative_news(title: str):
    """
    Return:
      (is_negative: bool, reason: str | None)
    """
    if not isinstance(title, str):
        return False, None

    t = title.lower()
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in t:
            return True, keyword
        
    return False, None

def get_sheet():
    """
    Authenticate and return Google Sheet worksheet
    """
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # Path credential (mounted by docker / github actions)
    CREDS_PATH = "service_account.json"

    if not os.path.exists(CREDS_PATH):
        raise FileNotFoundError("service_account.json not found")

    creds = Credentials.from_service_account_file(
        CREDS_PATH,
        scopes=SCOPES
    )

    client = gspread.authorize(creds)

    SPREADSHEET_NAME = "bank_news_scrapping_data"
    SHEET_NAME = "Sheet1"

    sheet = client.open(SPREADSHEET_NAME).worksheet(SHEET_NAME)
    return sheet

def get_existing_link_map(sheet):
    """
    Returns:
      dict: { link: row_number }
    """
    links = sheet.col_values(COL_LINK)  # kolom link
    return {
        link: idx + 1
        for idx, link in enumerate(links)
        if link
    }

def push_data():
    try:
        sheet = get_sheet()
    except Exception as e:
        print("Failed to connect to Google Sheet: ", e)
        return

    try:
        articles = scrape_google_news(limit=SCRAPING_LIMIT)
    except Exception as e:
        print("Scraping failed: ", e)
        return

    if not articles:
        print("No articles scraped")
        return

    existing_link_map = get_existing_link_map(sheet)
    existing_links = set(existing_link_map.keys())
    now = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    first_seen_at = now
    last_seen_at = now

    inserted = 0
    updated = 0
    skipped = 0

    for a in articles:
        link = a.get("link")
        title = a.get("title")
        published_at = a.get("published_at")

        if not link or not title:
            print("Skipping invalid article: ", a)
            skipped += 1
            continue

        if link in existing_links:
            row_idx = existing_link_map[link]
            try:
                sheet.update_cell(row_idx, COL_LAST_SEEN, last_seen_at)
                sheet.update_cell(row_idx, COL_PUBLISHED_AT, published_at)
                updated += 1
            except Exception as e:
                print("Failed to update published_at: ", e)
                skipped += 1
            continue

        is_negative, neg_keyword = check_negative_news(title)
        row = [
            first_seen_at,
            last_seen_at,
            published_at,
            a.get("year"),
            a.get("quarter"),
            a.get("source"),
            title,
            is_negative,
            neg_keyword,
            link
        ]

        try:
            sheet.append_row(row, value_input_option="USER_ENTERED")
            inserted += 1
            time.sleep(1) # to avoid hitting rate limits
        except Exception as e:
            print("Failed to append row: ", e)
            skipped += 1
            continue

    print(
        f"Job finished! | scraped={len(articles)} | inserted={inserted} | updated={updated} | skipped={skipped}"
    )

if __name__ == "__main__":
    push_data()