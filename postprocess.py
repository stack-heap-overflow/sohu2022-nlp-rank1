import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

def postprocess(prediction_file, raw_file, output_file):
    prediction = pickle.load(open(prediction_file, 'rb'))
    test_raw = list(map(json.loads, open(raw_file, "r+", encoding="utf-8").readlines()))
    f = open(output_file, 'w+', encoding="utf-8")
    f.write("id\tresult\n")
    step = 0
    for item in test_raw:
        output = {}
        for entity in item["entity"]:
            output[entity] = prediction[step] - 2
            step += 1
        f.write(f"{item['id']}\t{output}\n")
    f.close()
    assert len(prediction) == step

def get_text_feature(raw_file, feature_file, output_path):
    lines = list(map(json.loads, open(raw_file, "r+", encoding="utf-8").readlines()))
    features = pickle.load(open(feature_file, 'rb'))
    logits = np.array(features["logits"])
    print("Files Loaded")
    # text_features = np.zeros((len(lines), 768))
    item_id_list = []
    aggeration_feature_list = [f"sentiment_{agg}_{i}" for agg in ["mean", "max", "min", "std"] for i in range(5)]
    sentiment_features = np.zeros((len(lines), 5 * 4))
    step = 0
    for i in trange(len(lines)):
        line: dict = lines[i]
        item_id_list.append(line["id"])
        if not line["entity"] or not line["content"]:
            continue
        logits_part = logits[step:step+len(line["entity"])]
        sentiment_features[i, 0:5] = logits_part.mean(axis=0)
        sentiment_features[i, 5:10] = logits_part.max(axis=0)
        sentiment_features[i, 10:15] = logits_part.min(axis=0)
        sentiment_features[i, 15:20] = logits_part.std(axis=0)
        step += len(line["entity"])
    # assert step_1 == len(features["outputs"])
    assert step == len(features["logits"])
    output_table = pd.DataFrame()
    output_table["itemId"] = item_id_list
    output_table[aggeration_feature_list] = sentiment_features
    output_table.to_csv(os.path.join(output_path, "sentiment.csv"), index=False)
    output_table.to_feather(os.path.join(output_path, "sentiment.feather"))

if __name__ == "__main__":
    # output_path = "output"
    # postprocess(prediction_file=(output_file := os.path.join(output_path, "prediction.pkl")), raw_file="resources/nlp_data/test.txt", output_file=os.path.join(output_path, "section1_epoch2.txt"))
    output_path = "output_rec"
    get_text_feature(raw_file="resources/rec_data/recommend_content_entity_0317.txt", feature_file=os.path.join(output_path, "extra_output.pkl"), output_path=output_path)