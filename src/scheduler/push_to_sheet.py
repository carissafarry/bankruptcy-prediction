import os, re
from config import CONFIG
from datetime import datetime
from zoneinfo import ZoneInfo
from scraper.google_news import scrape_google_news

import gspread
from google.oauth2.service_account import Credentials


def check_negative_news(title: str):
    """
    Return:
      (is_negative: bool, reason: str | None)
    """
    if not isinstance(title, str):
        return False, None

    t = title.lower()
    for keyword in CONFIG["NEGATIVE_KEYWORDS"]:
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
    if not os.path.exists(CONFIG["GOOGLE_CREDS_PATH"]):
        raise FileNotFoundError("service_account.json not found")

    creds = Credentials.from_service_account_file(
        CONFIG["GOOGLE_CREDS_PATH"],
        scopes=SCOPES
    )

    client = gspread.authorize(creds)

    sheet = client.open(CONFIG["SPREADSHEET_NAME"]).worksheet(CONFIG["SHEET_NAME"])
    return client, sheet

def get_existing_link_map(sheet):
    """
    Returns:
      dict: { link: row_number }
    """
    links = sheet.col_values(CONFIG["COL_LINK"])  # kolom link
    return {
        link: idx + 1
        for idx, link in enumerate(links)
        if link
    }

def load_emiten_map(client):
    """
    Read Sheet2 and return emiten map.
    Expected columns:
      symbol | keywords
    """
    emiten_sheet = client.open(CONFIG["SPREADSHEET_NAME"]).worksheet("Sheet2")
    rows = emiten_sheet.get_all_records()

    emiten_map = {}
    for row in rows:
        code = row["symbol"].upper()
        keywords = [
            k.strip().lower()
            for k in row["keywords"].split(",")
            if k.strip()
        ]
        emiten_map[code] = keywords

    return emiten_map

def detect_emiten(article, emiten_map):
    text = f"{article.get('title','')} {article.get('source','')}".lower()

    for code, keywords in emiten_map.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return code
    return None

def col_to_a1(col_num: int) -> str:
    """
    1 -> A
    26 -> Z
    27 -> AA
    28 -> AB
    """
    result = ""
    while col_num > 0:
        col_num, remainder = divmod(col_num - 1, 26)
        result = chr(65 + remainder) + result
    return result

def push_data():
    try:
        client, sheet = get_sheet()
    except Exception as e:
        print("Failed to connect to Google Sheet: ", e)
        return

    try:
        articles = scrape_google_news(limit=CONFIG["SCRAPING_LIMIT"])
    except Exception as e:
        print("Scraping failed: ", e)
        return

    if not articles:
        print("No articles scraped")
        return

    emiten_map = load_emiten_map(client)
    existing_link_map = get_existing_link_map(sheet)
    existing_links = set(existing_link_map.keys())
    now = datetime.now(ZoneInfo(CONFIG["TIMEZONE"])).strftime("%Y-%m-%d %H:%M:%S")
    first_seen_at = now
    last_seen_at = now

    rows_to_insert = []
    updates = []
    
    inserted = 0
    updated = 0
    skipped = 0

    for a in articles:
        link = a.get("link")
        title = a.get("title")
        published_at = a.get("published_at")
        is_negative, neg_keyword = check_negative_news(title)
        emiten_code = detect_emiten(a, emiten_map)

        if not link or not title:
            print("Skipping invalid article: ", a)
            skipped += 1
            continue

        if link in existing_links:
            row_idx = existing_link_map[link]
            updates.append({
                "range": f"{col_to_a1(CONFIG['COL_LAST_SEEN'])}{row_idx}",
                "values": [[now]]
            })
            updates.append({
                "range": f"{col_to_a1(CONFIG['COL_PUBLISHED_AT'])}{row_idx}",
                "values": [[published_at]]
            })
            updates.append({
                "range": f"{col_to_a1(CONFIG['COL_SYMBOL'])}{row_idx}",
                "values": [[emiten_code]]
            })
            updated += 1
            continue

            # try:
            #     sheet.update_cell(row_idx, COL_LAST_SEEN, last_seen_at)
            #     sheet.update_cell(row_idx, COL_PUBLISHED_AT, published_at)
            #     sheet.update_cell(row_idx, COL_IS_NEGATIVE, is_negative)
            #     sheet.update_cell(row_idx, COL_NEG_KEYWORD, neg_keyword)
            #     updated += 1
            # except Exception as e:
            #     print("Failed to update published_at: ", e)
            #     skipped += 1

        rows_to_insert.append([
            first_seen_at,
            last_seen_at,
            published_at,
            a.get("year"),
            a.get("quarter"),
            a.get("source"),
            title,
            emiten_code,
            is_negative,
            neg_keyword,
            link
        ])
        inserted += 1

    insert_success = None
    if rows_to_insert:
        try:
            sheet.append_rows(
                rows_to_insert,
                value_input_option="USER_ENTERED"
            )
            insert_success = True
        except Exception as e:
            insert_success = False
            print("INSERT FAILED, SKIPPING UPDATE:", e)
            return

    if insert_success == None and updates:
        try:
            sheet.batch_update(updates)
        except Exception as e:
            print("UPDATE FAILED:", e)
            return
        
        # try:
        #     sheet.append_row(row, value_input_option="USER_ENTERED")
        #     inserted += 1
        #     time.sleep(1) # to avoid hitting rate limits
        # except Exception as e:
        #     print("Failed to append row: ", e)
        #     skipped += 1
        #     continue

    print(
        f"Job finished! | scraped={len(articles)} | inserted={inserted} | updated={updated if insert_success else 0} | skipped={skipped}"
    )

if __name__ == "__main__":
    push_data()