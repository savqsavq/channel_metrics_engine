import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import json
import random
from datetime import datetime

def generate_data(n=300, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    df = pd.DataFrame({
        "video_id": [f"vid_{i}" for i in range(n)],
        "views": np.random.randint(5000, 250000, n),
        "likes": np.random.randint(100, 12000, n),
        "subs_gained": np.random.randint(10, 500, n),
        "subs_lost": np.random.randint(0, 100, n),
        "duration_sec": np.random.uniform(60, 1200, n),
        "impressions": np.random.randint(20000, 500000, n),
        "ctr": np.random.uniform(1.5, 14.0, n),
        "revenue_usd": np.random.exponential(50, n),
        "avg_view_pct": np.random.uniform(15, 95, n),
        "watch_hours": np.random.uniform(5, 400, n),
        "shares": np.random.randint(0, 400, n),
        "comments": np.random.randint(0, 300, n),
        "title": [f"Video breakdown {i}" for i in range(n)]
    })
    return df

def feature_engineering(df):
    df["engagement_rate"] = (df.likes + df.comments + df.shares) / df.views
    df["log_views"] = np.log(df.views + 1)
    df["log_duration"] = np.log(df.duration_sec + 1)
    df["view_to_sub_ratio"] = df.views / (df.subs_gained + 1)
    df["title_length"] = df.title.apply(len)
    df["format_type"] = np.where(df.duration_sec < 300, "Short", "Long")
    df["net_subs"] = df.subs_gained - df.subs_lost
    df["ctr_per_minute"] = df.ctr / (df.duration_sec / 60)
    return df

def corr_matrix(df):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("correlation matrix")
    plt.tight_layout()
    plt.savefig("ctr_corr.png", dpi=120)
    plt.close()
    return corr

def fit_ols(df):
    cols = ["ctr","log_views","log_duration","likes","subs_gained",
            "revenue_usd","avg_view_pct","watch_hours","shares","comments",
            "engagement_rate","view_to_sub_ratio","title_length"]
    sub = df[cols].dropna()
    y = sub["ctr"]
    X = sm.add_constant(sub.drop(columns=["ctr"]))
    model = sm.OLS(y, X).fit()
    return model, sub

def fit_robust(df):
    cols = ["ctr","log_views","log_duration","likes","subs_gained",
            "revenue_usd","avg_view_pct","watch_hours","shares","comments",
            "engagement_rate","view_to_sub_ratio","title_length"]
    sub = df[cols].dropna()
    y = sub["ctr"]
    X = sm.add_constant(sub.drop(columns=["ctr"]))
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return model, sub

def model_summary(model):
    print(model.summary())
    coef = model.params.sort_values(key=lambda x: abs(x), ascending=False)
    top = coef.head(10)
    print("\ntop factors influencing ctr:\n")
    for k,v in top.items():
        print(f"{k:20} {round(v,4)}")

def residual_analysis(model, df):
    df["predicted_ctr"] = model.predict(sm.add_constant(df.drop(columns=["ctr"])))
    df["residual"] = df["ctr"] - df["predicted_ctr"]
    plt.scatter(df["predicted_ctr"], df["ctr"], c=df["residual"], cmap="bwr", alpha=0.6)
    plt.xlabel("predicted ctr")
    plt.ylabel("actual ctr")
    plt.title("predicted vs actual ctr")
    plt.colorbar(label="residual")
    plt.savefig("ctr_residuals.png", dpi=120)
    plt.close()
    return df

def top_bottom_videos(df, top_n=10):
    over = df.sort_values("residual", ascending=False).head(top_n)
    under = df.sort_values("residual").head(top_n)
    over["rank_type"] = "overperformer"
    under["rank_type"] = "underperformer"
    return pd.concat([over, under])

def split_by_format(df):
    short = df[df.format_type=="Short"]
    longf = df[df.format_type=="Long"]
    return short, longf

def run_analysis():
    data = generate_data()
    data = feature_engineering(data)
    print("rows:", len(data))
    corr = corr_matrix(data)
    corr.to_csv("corr_matrix.csv")
    short, longf = split_by_format(data)

    model_short, df_short = fit_robust(short)
    model_long, df_long = fit_robust(longf)

    print("\nshort format summary:")
    model_summary(model_short)
    print("\nlong format summary:")
    model_summary(model_long)

    df_short = residual_analysis(model_short, df_short)
    df_long = residual_analysis(model_long, df_long)

    combined = pd.concat([df_short.assign(format="Short"), df_long.assign(format="Long")])
    ranked = top_bottom_videos(combined)

    out_summary = {
        "timestamp": datetime.now().isoformat(),
        "short_mean_ctr": round(df_short.ctr.mean(), 2),
        "long_mean_ctr": round(df_long.ctr.mean(), 2),
        "corr_shape": corr.shape
    }
    with open("ctr_summary.json","w") as f:
        json.dump(out_summary, f, indent=2)

    ranked.to_csv("ctr_ranked_videos.csv", index=False)
    print("\nresults exported")

if __name__ == "__main__":
    run_analysis()





from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import textblob

def evaluate_model(df):
    y_true = df["ctr"]
    y_pred = df["predicted_ctr"]
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"rmse: {round(rmse,3)}")
    print(f"r2: {round(r2,3)}")
    return {"rmse": rmse, "r2": r2}

def detect_outliers(df, col="ctr", thresh=3.0):
    z = (df[col] - df[col].mean()) / df[col].std()
    df["outlier"] = (abs(z) > thresh).astype(int)
    print("outliers:", df.outlier.sum())
    return df

def variable_importance(model):
    coefs = model.params.drop("const", errors="ignore")
    imp = pd.DataFrame({
        "variable": coefs.index,
        "importance": np.abs(coefs.values)
    }).sort_values("importance", ascending=False)
    imp.to_csv("ctr_variable_importance.csv", index=False)
    plt.barh(imp.variable.head(15)[::-1], imp.importance.head(15)[::-1], color="#6699cc")
    plt.title("variable importance")
    plt.tight_layout()
    plt.savefig("ctr_importance.png", dpi=120)
    plt.close()
    return imp

def plot_relationships(df):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df.duration_sec, y=df.ctr, alpha=0.5)
    plt.title("ctr vs duration")
    plt.xlabel("duration (s)")
    plt.ylabel("ctr")
    plt.tight_layout()
    plt.savefig("ctr_vs_duration.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df.engagement_rate, y=df.ctr, alpha=0.5, color="orange")
    plt.title("ctr vs engagement rate")
    plt.tight_layout()
    plt.savefig("ctr_vs_engagement.png", dpi=120)
    plt.close()

def pca_view(df):
    num = df.select_dtypes(include=[np.number]).fillna(0)
    scaled = StandardScaler().fit_transform(num)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(scaled)
    plt.scatter(comps[:,0], comps[:,1], alpha=0.5, c=num["ctr"], cmap="plasma")
    plt.title("pca of numeric features")
    plt.savefig("ctr_pca.png", dpi=120)
    plt.close()

def title_sentiment(df):
    def score(text):
        try:
            return textblob.TextBlob(text).sentiment.polarity
        except:
            return 0
    df["sentiment"] = df.title.apply(score)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df.sentiment, y=df.ctr, alpha=0.5, color="#8844cc")
    plt.title("title sentiment vs ctr")
    plt.tight_layout()
    plt.savefig("ctr_vs_sentiment.png", dpi=120)
    plt.close()
    return df

def summarize_trends(df):
    trend = df.groupby("format_type").agg({
        "ctr":["mean","std"],
        "duration_sec":"mean",
        "views":"mean",
        "engagement_rate":"mean"
    })
    trend.columns = ["_".join(c) for c in trend.columns]
    trend.reset_index(inplace=True)
    print("\nformat summary:\n", trend)
    trend.to_csv("ctr_trends.csv", index=False)
    return trend

def top_keywords(df):
    from collections import Counter
    words = " ".join(df.title).lower().split()
    counts = Counter(words)
    common = pd.DataFrame(counts.most_common(25), columns=["word","freq"])
    plt.barh(common.word[::-1], common.freq[::-1], color="#779977")
    plt.title("common title words")
    plt.tight_layout()
    plt.savefig("ctr_common_words.png", dpi=120)
    plt.close()
    return common

def run_extended_analysis():
    data = generate_data()
    data = feature_engineering(data)
    data = detect_outliers(data)
    model, df_fit = fit_robust(data)
    df_fit = residual_analysis(model, df_fit)
    metrics = evaluate_model(df_fit)
    imp = variable_importance(model)
    plot_relationships(df_fit)
    pca_view(df_fit)
    df_fit = title_sentiment(df_fit)
    trends = summarize_trends(df_fit)
    words = top_keywords(df_fit)
    out = {
        "eval": metrics,
        "top_vars": imp.head(10).to_dict(orient="records"),
        "trends": trends.to_dict(orient="records"),
        "common_words": words.to_dict(orient="records"),
        "generated": datetime.now().isoformat()
    }
    with open("ctr_extended_report.json","w") as f:
        json.dump(out, f, indent=2)
    print("\nfull extended analysis complete")

if __name__ == "__main__":
    run_extended_analysis()