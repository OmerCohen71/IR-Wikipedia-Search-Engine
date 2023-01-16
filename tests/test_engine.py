import requests
import json
from time import time
r = requests.post('http://34.135.28.219:8080/get_pageview', json=[12,23,22092,278018])

r1 = requests.post('http://34.135.28.219:8080/get_pagerank', json=[3434750,25,12])

print(r.json())
print(r1.json())

########## CHECK METRICS HERE ##########

with open('queries_train.json', 'rt') as f:
  queries = json.load(f)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions)/len(precisions),3)


# url = 'http://35.232.59.3:8080'
# place the domain you got from ngrok or GCP IP below.
url = 'http://34.135.28.219:8080'

qs_res = []
for q, true_wids in queries.items():
    duration, ap = None, None
    t_start = time()
    try:
        res = requests.get(url + '/search', {'query': q}, timeout=35)
        duration = time() - t_start
        if res.status_code == 200:
            pred_wids, _ = zip(*res.json())
            ap = average_precision(true_wids, pred_wids)
            print(f"Retrival time for [{q}] with search -> {duration}\nAverage precision for [{q}] -> {ap}")
    except:
        pass

    qs_res.append((q, duration, ap))
