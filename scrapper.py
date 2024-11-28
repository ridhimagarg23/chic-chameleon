import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from selenium.common.exceptions import NoSuchWindowException, WebDriverException

# Pinterest Scraping
def pinterest_scrape(attributes, season, gender, category, type_of_outfit):
    query = "+".join([f"{key}:{value}" for key, value in attributes.items()])
    url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}&rs=typed"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve Pinterest data for {query}")
        return []
    
    soup = BeautifulSoup(response.content, "html.parser")
    items = []
    
    for pin in soup.find_all("div", class_="Yl- MIw HBf"):
        title = pin.find("div", class_="tBJ dyj").text if pin.find("div", class_="tBJ dyj") else "Unknown"
        price = pin.find("div", class_="rXtMi").text if pin.find("div", class_="rXtMi") else "N/A"
        items.append({"title": title, "price": price, "season": season})
    
    return items

# Myntra Scraping (Selenium-based)
def myntra_scrape(attributes, season, gender, category, type_of_outfit):
    query = "+".join([f"{key}:{value}" for key, value in attributes.items()])
    url = f"https://www.myntra.com/{gender}/{query.replace(' ', '-')}"
    
    # Setting up selenium with headless option
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
        time.sleep(5)  # Wait for the page to load
        
        # Scroll down to load more items (optional)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        items = []

        for product in soup.find_all("li", class_="product-base"):
            title = product.find("h3").text.strip() if product.find("h3") else "Unknown"
            price = product.find("span", class_="product-discountedPrice").text.strip() if product.find("span", class_="product-discountedPrice") else "N/A"
            items.append({"title": title, "price": price, "season": season})

    except (NoSuchWindowException, WebDriverException) as e:
        print(f"Error encountered with Myntra scraping: {e}")
        items = []

    finally:
        driver.quit()  # Ensure the driver is closed after use
    
    return items

# Amazon Scraping
def amazon_scrape(attributes, season, gender, category, type_of_outfit):
    query = "+".join([f"{key}:{value}" for key, value in attributes.items()])
    url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}&ref=nb_sb_noss"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve Amazon data for {query}")
        return []
    
    soup = BeautifulSoup(response.content, "html.parser")
    items = []
    
    for product in soup.find_all("div", class_="s-main-slot"):
        title = product.find("span", class_="a-text-normal").text.strip() if product.find("span", class_="a-text-normal") else "Unknown"
        price = product.find("span", class_="a-price-whole").text.strip() if product.find("span", class_="a-price-whole") else "N/A"
        items.append({"title": title, "price": price, "season": season})
    
    return items

# Ajio Scraping
def ajio_scrape(attributes, season, gender, category, type_of_outfit):
    query = "+".join([f"{key}:{value}" for key, value in attributes.items()])
    url = f"https://www.ajio.com/search/{query.replace(' ', '%20')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve Ajio data for {query}")
        return []
    
    soup = BeautifulSoup(response.content, "html.parser")
    items = []
    
    for product in soup.find_all("div", class_="item"):
        title = product.find("span", class_="title").text.strip() if product.find("span", class_="title") else "Unknown"
        price = product.find("span", class_="price").text.strip() if product.find("span", class_="price") else "N/A"
        items.append({"title": title, "price": price, "season": season})
    
    return items

# Main function to use these scraping functions based on input
def main(attributes, season, gender, category, type_of_outfit):
    pinterest_data = pinterest_scrape(attributes, season, gender, category, type_of_outfit)
    myntra_data = myntra_scrape(attributes, season, gender, category, type_of_outfit)
    amazon_data = amazon_scrape(attributes, season, gender, category, type_of_outfit)
    ajio_data = ajio_scrape(attributes, season, gender, category, type_of_outfit)
    
    # Combining data from all sources
    all_data = pinterest_data + myntra_data + amazon_data + ajio_data
    
    # Returning combined data
    return all_data

# Example usage
attributes = {
    'color': 'red',
    'size': 'M',
    'material': 'cotton'
}
season = 'Summer'
gender = 'Men'
category = 'T-shirt'
type_of_outfit = 'Casual'

data = main(attributes, season, gender, category, type_of_outfit)
for item in data:
    print(item)
