import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://news.google.com"
SEARCH_URL = (
    "https://news.google.com/search"
    "?q=bank%20when%3A1y&hl=id&gl=ID&ceid=ID%3Aid" # 1Y before
)
TIMEOUT_REQUEST = 15

def parse_published_at(time_tag):
    """
    Parse <time datetime="2025-08-11T07:00:00Z">
    Return: timezone-aware datetime (Asia/Jakarta)
    """
    if not time_tag:
        return None

    dt_raw = time_tag.get("datetime")
    if not dt_raw:
        return None

    # Convert Z (UTC) → WIB
    dt_utc = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
    return dt_utc.astimezone(ZoneInfo("Asia/Jakarta"))

def get_year_and_quarter(dt: datetime):
    quarter = (dt.month - 1) // 3 + 1
    return dt.year, quarter

def scrape_google_news(limit=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MVP-Scraper/1.0)"
    }

    resp = requests.get(SEARCH_URL, headers=headers, timeout=TIMEOUT_REQUEST)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    articles = []
    cards = soup.select("c-wiz.PO9Zff")
    print(f"Found cards: {len(cards)}")

    for card in cards:
        # title + link
        title_tag = card.select_one("a.JtKRv")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        href = title_tag.get("href")

        if not title or not href:
            continue

        link = urljoin(BASE_URL, href.replace("./", ""))

        # source
        source_tag = card.select_one(".vr1PYe")
        source = source_tag.get_text(strip=True) if source_tag else None

        # published time
        time_tag = card.select_one("time.hvbAAd")
        published_at = parse_published_at(time_tag)
        if not published_at:
            continue 

        year, quarter = get_year_and_quarter(published_at)

        articles.append({
            "published_at": published_at.strftime("%Y-%m-%d %H:%M:%S"),
            "year": year,
            "quarter": quarter,
            "source": source,
            "title": title,
            "link": link,
        })

        if len(articles) >= limit:
            break

    return articles


if __name__ == "__main__":
    data = scrape_google_news(limit=5)
    for d in data:
        print(d)