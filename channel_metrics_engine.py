import pandas as pd
import numpy as np

def build_mock_metrics(n=20):
    df = pd.DataFrame({
        "video_id": [f"vid_{i}" for i in range(n)],
        "views": np.random.randint(5_000, 250_000, n),
        "likes": np.random.randint(100, 10_000, n),
        "avg_watch": np.random.uniform(0.2, 0.9, n),
        "ctr": np.random.uniform(2.5, 12.0, n)
    })
    df["engagement_score"] = (df.likes / df.views) + df.avg_watch
    return df

if __name__ == "__main__":
    data = build_mock_metrics()
    print(data.head())