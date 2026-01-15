import os
from datetime import datetime, time
from scraper.google_news import scrape_google_news

import gspread
from google.oauth2.service_account import Credentials


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
    links = sheet.col_values(5)  # kolom link
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
        articles = scrape_google_news(limit=5)
    except Exception as e:
        print("Scraping failed: ", e)
        return

    if not articles:
        print("No articles scraped")
        return

    existing_links = set(sheet.col_values(5))
    existing_link_map = get_existing_link_map(sheet)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    inserted = 0
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
                sheet.update_cell(row_idx, 4, published_at)
                skipped += 1
            except Exception as e:
                print("Failed to update published_at: ", e)
                skipped += 1
            continue

        row = [
            now,
            title,
            a.get("source"),
            published_at,
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
        f"Job finished! | inserted={inserted} | skipped={skipped} | scraped={len(articles)}"
    )

if __name__ == "__main__":
    push_data()