import json
import pandas as pd

# получить json для тестового предсказания
def get_sample_json(wine_type, n):
    df = pd.read_csv(f'data/winequality-{wine_type}.csv', sep=';')
    df = df.drop('quality', axis=1)
    result = df.sample(n).to_json()
    return result


#print(get_sample_json('red', 5))





