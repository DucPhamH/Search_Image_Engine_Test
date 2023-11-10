import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path


app = Flask(__name__)

fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
feature_path = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]
        print(scores)

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
