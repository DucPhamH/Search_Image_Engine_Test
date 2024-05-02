from io import BytesIO
from PIL import Image
from pathlib import Path
import numpy as np 
import requests

from feature_extractor import FeatureExtractor
from pymongo import MongoClient

MONGO_URI = 'mongodb+srv://ducphamhong2:05042001d@cookhealthy.qzhvhn8.mongodb.net/'
DATABASE_NAME = 'test'
COLLECTION_NAME = 'test_search'

#connect to mongodb
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# viết hàm tải ảnh từ link lưu trên mongodb và lưu vào thư mục static/img
def download_img():
    searchs = list(collection.find())
    print(searchs)
    for search in searchs:
        img_url = search['img']  # Đường dẫn của hình ảnh từ MongoDB
        img_name = search['img_name']  # Tên của hình ảnh từ MongoDB
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img.save(f'static/img/{img_name}.jpg')
        



if __name__ == "__main__":
    fe = FeatureExtractor()
    download_img()
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)

        feature = fe.extract(img=Image.open(img_path))
        # print(type(feature), feature.shape)
        print(feature)

        feature_path = Path("./static/feature") / (img_path.stem + ".npy")
        print(feature_path)

        np.save(feature_path, feature)