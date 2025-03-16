import time
import pandas as pd
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

ptt_boards = {
    "baseball": "https://www.ptt.cc/bbs/baseball/index.html",
    "boy-girl": "https://www.ptt.cc/bbs/Boy-Girl/index.html",
    "c_chat": "https://www.ptt.cc/bbs/c_chat/index.html",
    "hatepolitics": "https://www.ptt.cc/bbs/hatepolitics/index.html",
    "lifeismoney": "https://www.ptt.cc/bbs/Lifeismoney/index.html",
    "military": "https://www.ptt.cc/bbs/Military/index.html",
    "pc_shopping": "https://www.ptt.cc/bbs/pc_shopping/index.html",
    "stock": "https://www.ptt.cc/bbs/stock/index.html",
    "tech_job": "https://www.ptt.cc/bbs/Tech_Job/index.html",
}

MAX_ARTICLES = 200000

def crawl_board(board_url, board_name):
    print(f"Start crawling {board_name} ...")

    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service('/Users/bella/.wdm/drivers/chromedriver/mac64/134.0.6998.88/chromedriver-mac-arm64/chromedriver')
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(board_url)

    try:
        # Pass age button
        age_button = driver.find_element(By.XPATH, "//button[contains(text(), '我同意')]")
        age_button.click()
    except NoSuchElementException:
        pass

    data = []

    while len(data) < MAX_ARTICLES:
        articles = driver.find_elements(By.CSS_SELECTOR, ".r-ent")

        for article in articles:
            # Get titles and links
            try:
                title_tag = article.find_element(By.CSS_SELECTOR, ".title a")
                title = title_tag.text
                link = title_tag.get_attribute("href")
            except NoSuchElementException:
                continue
            # Get authors
            author_tag = article.find_element(By.CSS_SELECTOR, ".meta .author")
            author = author_tag.text
            # Get dates
            date_tag = article.find_element(By.CSS_SELECTOR, ".meta .date")
            date = date_tag.text

            # Get push numbers
            push_tag = article.find_elements(By.CSS_SELECTOR, ".nrec span")
            if push_tag:
                push_count = push_tag[0].text.strip()
                if push_count == "爆":
                    push_count = "100"
                elif push_count == "XX":
                    push_count = "-100"
                elif push_count.startswith("X"):
                    push_count = f"-{push_count[1:]}"
            else:
                push_count = "0"

            data.append((link, title, author, date, push_count))

            if len(data) >= MAX_ARTICLES:
                break

        # Go to previous pages
        try:
            prev_page = driver.find_element(By.LINK_TEXT, "‹ 上頁")
            prev_page_url = prev_page.get_attribute("href")
            print(f"{board_name}: Prev {prev_page_url}")
            driver.get(prev_page_url)
            time.sleep(1)
        except NoSuchElementException:
            print(f"{board_name}: no more pages.")
            break

    driver.quit()

    df = pd.DataFrame(data, columns=["Title", "Link", "Author", "Date", "Push_Count"])
    file_name = f"{board_name}-titles.csv"
    df.to_csv(file_name, index=False, encoding="utf-8-sig")
    print(f"{board_name} done!")

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(crawl_board, url, board) for board, url in ptt_boards.items()]
    concurrent.futures.wait(futures)

print("All CSV files have been stored.")