import json
import ssl
from urllib.request import urlopen

# Constants
BASE_URL = "https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid=DSAA31&attr=&pageCount=40"
OUTPUT_PRODUCTS_FILE = "products.txt"
OUTPUT_BEST_PRODUCTS_FILE = "best-products.txt"
OUTPUT_STANDARDIZATION_FILE = "standardization.csv"

# Create an SSL context that ignores certificate verification
context = ssl._create_unverified_context()

def fetch_data(page: int):
    """Fetch product data from the API for a given page."""
    url = f"{BASE_URL}&page={page}"
    response = urlopen(url, context=context)
    return json.loads(response.read().decode('utf-8'))

def gather_product_data(total_pages: int):
    """Gather product IDs, best products, i5 processor data, and prices."""
    product_ids = []
    best_products = []
    i5_processors_price = 0
    i5_processors_count = 0
    prices = []

    for page in range(1, total_pages + 1):
        data = fetch_data(page)
        products = data['Prods']
        
        for product in products:
            product_id = product['Id']
            price = product['Price']
            product_name = product['Name']
            review_count = product['reviewCount'] if product['reviewCount'] else 0
            rating_value = product['ratingValue'] if product['ratingValue'] else 0

            product_ids.append(product_id)
            prices.append(price)

            # Identify best products
            if review_count > 1 and rating_value > 4.9:
                best_products.append(product_id)

            # Collect i5 processor data
            if "i5" in product_name:
                i5_processors_price += price
                i5_processors_count += 1

    return product_ids, best_products, i5_processors_price, i5_processors_count, prices

def write_to_file(filename: str, data: list):
    """Write a set of data to a file, one item per line."""
    with open(filename, "w") as file:
        file.writelines(f"{item}\n" for item in data)

def calculate_z_scores(prices: list):
    """Calculate the z-scores for the given list of prices."""
    mean_price = sum(prices) / len(prices)
    std_dev_price = (sum((x - mean_price) ** 2 for x in prices) / len(prices)) ** 0.5
    return [(price - mean_price) / std_dev_price for price in prices]

def write_standardization_file(product_ids: list, prices: list, z_scores: list):
    """Write ProductID, Price, and PriceZScore to a CSV file."""
    with open(OUTPUT_STANDARDIZATION_FILE, "w") as file:
        file.write("ProductID,Price,PriceZScore\n")
        for product_id, price, z_score in zip(product_ids, prices, z_scores):
            file.write(f"{product_id},{price},{z_score}\n")

def main():
    """Main function to orchestrate the workflow."""
    # Fetch initial data to get the total number of pages
    initial_data = fetch_data(1)
    total_pages = initial_data['TotalPage']

    # Gather product data
    product_ids, best_products, i5_processors_price, i5_processors_count, prices = gather_product_data(total_pages)

    # Write product IDs and best products to files
    write_to_file(OUTPUT_PRODUCTS_FILE, product_ids)
    write_to_file(OUTPUT_BEST_PRODUCTS_FILE, best_products)

    # Print average price of i5 processors
    average_i5_price = i5_processors_price / i5_processors_count
    print(average_i5_price)

    # Calculate z-scores and write to CSV
    z_scores = calculate_z_scores(prices)
    write_standardization_file(product_ids, prices, z_scores)

if __name__ == "__main__":
    main()
