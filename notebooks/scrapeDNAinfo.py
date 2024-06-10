import time
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

#df = pd.read_csv(city_list)

def airdna_scraper(cities) :
    
    opts = FirefoxOptions()
    # opts.add_argument("--headless")

    driver = webdriver.Firefox(options=opts)
    driver.get("https://app.airdna.co/data/us?tab=markets")


    area_name_class = "MuiTypography-root MuiTypography-titleXXS css-1l2kcxs"

    listing_info_class = "MuiTypography-root MuiTypography-titleXXS css-3ry9xp"




    # area_names = driver.find_elements(By.CLASS_NAME, area_name_class)

    listing_data = []

    listing_class = "MuiTypography-root MuiTypography-inherit MuiLink-root MuiLink-underlineAlways css-7g648y"

    WebDriverWait(driver, 60).until(
        EC.visibility_of_all_elements_located((By.CLASS_NAME, listing_class))
    )

    listings_captured = 0
    max_listings = 3


    while listings_captured < max_listings:

        listings = driver.find_elements(By.CLASS_NAME, listing_class)
        current_listing_count = len(listings)

        if current_listing_count <= listings_captured:
            break  # No new reviews are being loaded

        for listing in listings[listings_captured:]:
            if listings_captured >= max_listings:
                break
            
            driver.execute_script("arguments[0].scrollIntoView();", listing)
        
            try:
                # Extracting the relevant data from each review element
                name = listing.find_element(By.CLASS_NAME, area_name_class).text
                info = listing.find_element(By.CLASS_NAME, listing_info_class).text

                if name in cities:

                # Collecting the data
                    listing_data.append({
                        "name": name,
                        "info": info,
                    })

                    listings_captured_captured += 1

            except Exception as e:
                print(f'Could not extract listing data: {e}')
                continue

        # Scroll down to load more reviews if needed
        if listings_captured < max_listings:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            WebDriverWait(driver, 20).until(
                EC.visibility_of_all_elements_located((By.CLASS_NAME, area_name_class)))

    driver.quit()

    return listing_data


city_list = ['New York','Birmingham','Knoxville','Montgomery','New Haven','San Diego','Helena','St. Louis','Chattanooga','Louisville','Lexington']

airdna_scraper(city_list)