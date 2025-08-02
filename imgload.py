import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote

# Load your CSV file
csv_path = "C:/project sem 6/DL PROJECT/Real-TIme-Skin-Type-Detection-main/bruhhh/skincare_products.csv"
df = pd.read_csv(csv_path)

# Folder to save images
os.makedirs("product_images", exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Store names of products that downloaded successfully
successful_downloads = []

def download_image(product_name):
    search_url = f"https://www.bing.com/images/search?q={quote(product_name)}"
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    
    img_tags = soup.find_all("img")
    for img in img_tags:
        img_url = img.get("src")
        if img_url and img_url.startswith("http"):
            try:
                img_data = requests.get(img_url).content
                file_path = f"product_images/{product_name[:50]}.jpg"
                with open(file_path, "wb") as handler:
                    handler.write(img_data)
                print(f"✅ Downloaded: {product_name}")
                successful_downloads.append(product_name)
                return
            except Exception as e:
                print(f"❌ Error downloading {product_name}: {e}")
                return
    print(f"⚠️ No image found for: {product_name}")

# Try downloading each product image
for product in df["product_name"]:
    download_image(product)

# Keep only the rows with successful downloads
filtered_df = df[df["product_name"].isin(successful_downloads)]

# Overwrite the CSV with filtered rows
filtered_df.to_csv(csv_path, index=False)
print("\n✅ CSV updated with only successfully downloaded products.")
