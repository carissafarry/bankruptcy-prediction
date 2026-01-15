import os
from datetime import datetime

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


def push_data():
    sheet = get_sheet()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = "Hello from scheduled Docker job"

    row = [
        timestamp,
        message
    ]

    sheet.append_row(row, value_input_option="USER_ENTERED")

    print("Successfully pushed data to Google Sheet")
    print("➡ Row:", row)


if __name__ == "__main__":
    push_data()