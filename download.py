from datasets import load_dataset
import pandas as pd

# 下载
dataset = load_dataset("ag_news", split="train")

# 保存到本地
df = pd.DataFrame({'text': dataset['text'], 'label': dataset['label']})
df.to_csv('./data/raw/ag_news/ag_news_train.csv', index=False)

print("AG News 下载成功！")