from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


import time
import pandas as pd

import os

from google.cloud import storage

df = pd.read_csv(r"/home/castillosam27/city_urls.csv")
url_list = list(df.url)

def insert_reviews_path(city_url):
    """Add the string '/reviews' to the appropriate place in an Airbnb listing URL. 
    The new URL navigates to the reviews page of that listing."""
    parsed_url = urlparse(city_url)
    query_params = parse_qs(parsed_url.query)
    new_path = parsed_url.path + "/reviews"
    query_params_str = urlencode(query_params, doseq=True)
    new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, new_path, parsed_url.params, query_params_str, parsed_url.fragment))
    return new_url

def collect_review_urls(city_url, max_pages=5):
    """ 
    Initiates a Selenium webdriver in Chrome that opens an Airbnb URL of listings in a certain city;
    navigates to each listing on the page (20 total) and collects URLs of specific listings;
    uses insert_reviews_path function defined above to transform listing URLs into review page URLs;
    stores and returns these review page URLs as 'review_urls' list.
    """
    listings_class = "itu7ddv"

    opts = FirefoxOptions()
    opts.add_argument("--headless")

    driver = webdriver.Firefox(options=opts)
    driver.get(city_url)

    review_urls = []
    current_page = 1
    
    print(f"Total pages to scan: {max_pages}.")
    
    while current_page <= max_pages:
        WebDriverWait(driver, 60).until(
            EC.visibility_of_all_elements_located((By.CLASS_NAME, listings_class)))
        
        listings = driver.find_elements(By.CLASS_NAME, listings_class)
        
        status = 'Not ready - detecting more listings. . .'
        
        while status == 'Not ready - detecting more listings. . .':
            if len(listings) < 20:
                status = 'Not ready - detecting more listings. . .'
                WebDriverWait(driver, 60).until(
                    EC.visibility_of_all_elements_located((By.CLASS_NAME, listings_class)))
                listings = driver.find_elements(By.CLASS_NAME, listings_class)
                print(status)
                
            else:
                status = f"Ready! {len(listings)} total listings detected on page {current_page}."
                print(status)
                
                if len(listings) > 20:
                    print(f"Estimated {len(listings) - 20} are not valid elements - expect hiccups.")
                    
        for listing in listings:
            try:
                driver.execute_script("arguments[0].scrollIntoView();", listing)
                driver.execute_script("arguments[0].click();", listing)
                
                original_window = driver.current_window_handle
                WebDriverWait(driver, 20).until(EC.number_of_windows_to_be(2))
                handles = driver.window_handles
                driver.switch_to.window(handles[-1])
                
                WebDriverWait(driver, 10).until(EC.url_contains("airbnb.com/rooms/"))
                
                current_url = driver.current_url
                review_url = insert_reviews_path(current_url)
                review_urls.append(review_url)

                driver.close()
                driver.switch_to.window(original_window)
                print(f"Listing captured successfully: {review_url[:50]}...")
                
            except Exception as e:
                print(f'Could not access listing: {listing}. Exception: {e}')
                continue
        
        # Move to the next page
        if current_page < max_pages:
            try:
                next_page_xpath = "//*[@aria-label='Next']"
                next_button = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, next_page_xpath)))
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                next_button.click()
                print(f"Moving to page {current_page + 1} . . .")
                time.sleep(2)
            except Exception as e:
                print(f'Could not navigate to next page: {e}')
                break
        
        current_page += 1
    
    driver.quit()
    return review_urls

def review_scraper(review_url):
    """Scrapes reviews from a given Airbnb review URL."""

    opts = FirefoxOptions()
    opts.add_argument("--headless")

    driver = webdriver.Firefox(options=opts)
    driver.get(city_url)

    review_data = []
    
    try:
        driver.get(review_url)

        review_class = "r1are2x1"

        WebDriverWait(driver, 60).until(
            EC.visibility_of_all_elements_located((By.CLASS_NAME, review_class)))

        reviews_captured = 0
        max_reviews = 200
        

        while reviews_captured < max_reviews:

            reviews = driver.find_elements(By.CLASS_NAME, review_class)
            current_review_count = len(reviews)

            if current_review_count <= reviews_captured:
                break  # No new reviews are being loaded

            for review in reviews[reviews_captured:]:
                if reviews_captured >= max_reviews:
                    break

                driver.execute_script("arguments[0].scrollIntoView();", review)

                try:
                    # Extracting the relevant data from each review element
                    name = review.find_element(By.CLASS_NAME, "hpipapi").text
                    rating_info = review.find_element(By.CLASS_NAME, "s78n3tv").text
                    review_text = review.find_element(By.CLASS_NAME, "r1bctolv").text

                    # Collecting the data
                    review_data.append({
                        "name": name,
                        "rating_info": rating_info,
                        "review_text": review_text
                    })

                    reviews_captured += 1

                except Exception as e:
                    print(f'Could not extract review data: {e}')
                    continue

            # Scroll down to load more reviews if needed
            if reviews_captured < max_reviews:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                WebDriverWait(driver, 20).until(
                    EC.visibility_of_all_elements_located((By.CLASS_NAME, review_class)))

    except Exception as e:
        print(f"An error occurred while scraping reviews from {review_url}: {e}")
        # Log the error or take other appropriate actions
    
    finally:
        driver.quit()
        
    return review_data

def extract_city_from_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the city from the path
    # The city is the first segment after '/s/' in the path
    path_segments = parsed_url.path.split('/')
    
    # Ensure the path structure is as expected
    if len(path_segments) >= 3 and path_segments[1] == 's':
        city = path_segments[2]
        return city
    else:
        raise ValueError("URL structure is not as expected")
    
def scrapeBNBrevs(city_url):
    
    city_name = extract_city_from_url(city_url)
    
    print(f"Collecting Review URLs for {city_name}. . .")
    
    review_urls = set(collect_review_urls(city_url, max_pages=5))
        
    reviews = []
    max_reviews = 2000
    
    print(f"Capturing reviews for {city_name}. . .")
    for idx, url in enumerate(review_urls):
        
        if len(reviews) >= max_reviews:
            break
        
        print(f"Scanning {idx} / 100 potential listings.")
        reviews.extend(review_scraper(url))
        print(f'Updated review count for {city_name}: {len(reviews)} / {max_reviews} max.')

        
    # Filter out duplicate reviews
        
    seen = set()
    unique_reviews = [d for d in reviews if tuple(d.items()) not in seen and not seen.add(tuple(d.items()))]
    print(f'Total unique reviews for {city_name}: {len(unique_reviews)}.')    
    
    review_df = pd.DataFrame(unique_reviews, columns=unique_reviews[0].keys())
    
    return review_df

# Initialize the Google Cloud Storage client
# storage_client = storage.Client()

# # Specify the name of your Cloud Storage bucket
# bucket_name = "investmentdata"

# # Iterate over each city URL
# for idx, city_url in enumerate(url_list):
#     city_name = extract_city_from_url(city_url)
    
#     print('*'*50)
#     print(f"{city_name}, {idx} / {len(url_list)} total cities.")
#     print('*'*50)
    
#     # Scrape reviews for the current city
#     city_reviews_df = scrapeBNBrevs(city_url)
    
#     # Specify the filename within the bucket
#     filename = f"{city_name}_reviews.csv"
    
#     # Write the DataFrame to a CSV file-like object
#     csv_data = city_reviews_df.to_csv(index=False)
    
#     # Specify the path to the file within the bucket
#     blob_path = f"city_reviews/{filename}"
    
#     # Upload the file to Cloud Storage
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(blob_path)
#     blob.upload_from_string(csv_data, content_type='text/csv')
    
#     print('*'*50)
#     print(f"Saved {city_name} reviews to file: gs://{bucket_name}/{blob_path}")
#     print('*'*50)

def save_to_local(directory, city_name, city_reviews_df):
    """Save DataFrame to a local directory on the VM."""
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Construct the file path
    filename = f"{directory}/{city_name}_reviews.csv"

    # Save the DataFrame to a CSV file
    city_reviews_df.to_csv(filename, index=False)

    print(f"Saved {city_name} reviews to file at {filename}")

# Example usage in your main script
directory = "/home/castillosam27/city_reviews"
for idx, city_url in enumerate(url_list):
    city_name = extract_city_from_url(city_url)
    print(f"Collecting reviews for {city_name}... ({idx + 1} / {len(url_list)})")

    # Scrape reviews for the current city
    city_reviews_df = scrapeBNBrevs(city_url)

    # Save to local directory
    save_to_local(directory, city_name, city_reviews_df)
