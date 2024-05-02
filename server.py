import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from pathlib import Path
from flask_cors import CORS
from pymongo import MongoClient


app = Flask(__name__)
CORS(app)

fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    print(feature_path)
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
feature_path = np.array(features)

# // in ra feature_path để xem kết quả


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)
        print(dists)
        ids = np.argsort(dists)[:30]
        print(ids)
        scores = [(dists[id], img_paths[id]) for id in ids]
        print(scores)

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)


#//////////////////////////

# app = Flask(__name__)
# CORS(app)

# MONGO_URI = 'mongodb+srv://ducphamhong2:05042001d@cookhealthy.qzhvhn8.mongodb.net/'
# DATABASE_NAME = 'test'
# COLLECTION_NAME = 'test_search'

# #connect to mongodb
# client = MongoClient(MONGO_URI)
# db = client[DATABASE_NAME]
# collection = db[COLLECTION_NAME]


# fe = FeatureExtractor()
# features = []
# img_paths = []
# for feature_path in Path("./static/feature").glob("*.npy"):
#     print(feature_path)
#     features.append(np.load(feature_path))
#     img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
# feature_path = np.array(features)

# # // in ra feature_path để xem kết quả
# print(features)
# print(img_paths)




#/////////////////////
# @app.route('/search', methods=['POST'])
# def search():
#     if 'query_img' not in request.files:
#         return jsonify({'error': 'No query image found in request'}), 400

#     file = request.files['query_img']
#     img = Image.open(file.stream)  # PIL image
#     query = fe.extract(img)
#     dists = np.linalg.norm(features - query, axis=1)
#     ids = np.argsort(dists)[:30]
#     # scores = [{'distance': float(dists[id]), 'image_path': str(img_paths[id])} for id in ids]
#     # in ra tên ảnh và distance

#     scores = [
#         {
#             'distance': float(dists[id]),
#             # in ra tên ảnh loại bỏ đường dẫn, nếu là windows thì split theo dấu \\ còn nếu là linux thì split theo dấu /
#             # xóa cả đuôi định dạng ảnh
#             'image_path': str(img_paths[id]).split('\\')[-1].split('/')[-1].split('.')[0]
#         }
#         for id in ids
#     ]

#     # tìm kiếm trong db sao cho tên ảnh trong scores trùng với img_name trong db và sắp xếp theo distance
    
#     for score in scores:
#         search = collection.find_one({'img_name': score['image_path']})
#         score['_id'] = str(search['_id'])
#         score['img'] = search['img']
#         score['img_name'] = search['img_name']

#     return jsonify({'scores': scores}), 200
#     # return jsonify({'scores': scores}), 200

# if __name__ == '__main__':
#     app.run(debug=True)
