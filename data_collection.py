import os
import urllib.request
import tarfile

# 1. Download the dataset
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
urllib.request.urlretrieve(url, "aclImdb_v1.tar.gz")

# 2. Extract the archive
with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
    tar.extractall()

# 3. Read the text files into memory
def load_reviews(split_dir):
    texts, labels = [], []
    for sentiment in ['pos', 'neg']:
        folder = os.path.join("aclImdb", split_dir, sentiment)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                # 1 for positive, 0 for negative
                labels.append(1 if sentiment == 'pos' else 0)
    return texts, labels

train_texts, train_labels = load_reviews('train')
test_texts, test_labels = load_reviews('test')