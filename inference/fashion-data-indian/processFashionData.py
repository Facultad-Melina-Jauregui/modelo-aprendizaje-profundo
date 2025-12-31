import zipfile
import pandas as pd 
import os

def process_image(image_str):
    if '~^' in image_str:
        images = image_str.split("~^")
        if images[0] == "":
            return images[1]
        return images[0]
    else:
        return image_str.split(",")[0].strip('"')



# ---- extract from original csv file  ----
zip_path = "Fashion Data.csv.zip"
csv_filename = "Fashion Data.csv"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract(csv_filename)

df = pd.read_csv(csv_filename)

columns_to_drop = ["date_stored", "star_rating", "meta_data", "size", "mrp", "product_code", "competitor", "set_product_price", "colour"]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

required_columns = ["uuid", "url", "title", "brand", "images", "selling_price", "product_detials", "category"]
df.dropna(subset=required_columns, inplace=True)

df["images"] = df["images"].apply(process_image)
df = df.rename(columns={"images": "image"})
df["product_detials"] = df.apply(lambda row: f'{row["category"]}: {row["product_detials"]}', axis=1)
df.drop(columns=["category"], inplace=True)
df = df.rename(columns={"product_detials": "product_details"})

df.dropna(subset=["image", "product_details"], inplace=True)
processed_csv = "fashion_data.csv"
df.to_csv(processed_csv, index=False)
os.remove(csv_filename)
