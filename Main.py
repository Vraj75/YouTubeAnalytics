#!/usr/bin/env python3
"""
Complete PySpark mini-project 

Features included :
-  RDD operations, PairRDDs, map/reduce (wordcount), joins (example), numeric transforms,
          broadcast variables, accumulators, partitioning, persist/cache, checkpointing.

- U sampling, calling external program, Spark configuration printout, optimization tips,
          simple MLlib ALS recommender example.

Outputs:
- ./outputs/cleaned.parquet
- ./outputs/top_categories.csv
- ./outputs/best_hours.csv
- ./outputs/top_videos.csv
- ./outputs/trending_keywords_overall.csv
"""
import os
import re
import json
import subprocess
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, to_timestamp, expr, lower, split, explode, hour, date_format,
    sum as spark_sum, avg as spark_avg, count as spark_count, desc, row_number,
    size, regexp_replace
)
from pyspark.sql.window import Window

# ---------- Config ----------
DATASET_PATH = "/Users/vraj/Desktop/YoutubeAnalytics/USvideos.csv"
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# fallback category mapping (used as broadcast variable)
FALLBACK_CAT_MAP = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
    17: "Sports", 19: "Travel & Events", 20: "Gaming", 22: "People & Blogs",
    23: "Comedy", 24: "Entertainment", 25: "News & Politics", 26: "Howto & Style",
    27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism", 43: "Shows"
}

REPORT_FILE = "/mnt/data/bda assign.docx"  # reference to uploaded report file (if needed)

# ---------- Utility functions ----------
def create_spark(app_name="YouTubeBDA", master="local[*]"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    # checkpoint directory
    ckpt = os.path.abspath("./spark_checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    spark.sparkContext.setCheckpointDir(ckpt)
    return spark

def safe_read_csv(spark, path):
    """Read CSV robustly (multiLine and escape handling)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Update DATASET_PATH.")
    # Read without inferSchema to avoid schema inference exceptions; we'll cast later.
    df = spark.read.option("header", True).option("multiLine", True).option("escape", "\"").csv(path)
    return df

# ---------- RDD basics & PairRDDs ----------
def rdd_wordcount_demo(sc, df):
    """
    Demonstrates RDD transformations and actions:
     - Create an RDD of titles (text), flatMap -> words, map -> (word,1) -> reduceByKey -> counts
    """
    print("\n--- RDD wordcount (titles) ---")
    # Convert titles column to an RDD of lines (drop nulls)
    titles_rdd = sc.parallelize(df.select("title").rdd.map(lambda r: r[0] if r and r[0] else "") \
                                 .filter(lambda s: s != "").collect())
    # If dataset is large, better use titles = df.select("title").rdd.map(...)
    # We'll demonstrate both: small sample via collect above; also show direct RDD ops using df.rdd
    # Direct RDD approach (scalable):
    direct_titles_rdd = df.rdd.map(lambda r: r['title'] if r['title'] else "")
    words = direct_titles_rdd.flatMap(lambda line: re.split(r'\W+', line.lower())).filter(lambda w: w and len(w)>1)
    word_counts = words.map(lambda w: (w, 1)).reduceByKey(lambda a,b: a+b)
    # show top 10
    top10 = word_counts.takeOrdered(10, key=lambda x: -x[1])
    print("Top 10 words in titles:", top10[:10])
    return word_counts

def pair_rdd_join_demo(sc):
    """
    Small synthetic pairRDD join demo to demonstrate PairRDD joins (UNIT-3).
    Creates two small RDDs and performs join / leftOuterJoin.
    """
    print("\n--- PairRDD join demo ---")
    users = sc.parallelize([("u1","Alice"), ("u2","Bob"), ("u3","Charlie")])
    events = sc.parallelize([("u1", ("watch", "2025-01-01")), ("u2", ("like", "2025-01-02"))])
    joined = users.join(events)  # inner join
    left = users.leftOuterJoin(events)
    print("Joined sample:", joined.collect())
    print("Left join sample:", left.collect())
    return joined, left

# ---------- Preprocessing & cleaning (DataFrame style) ----------
def preprocess_df(df):
    """
    Clean & cast key columns. Handles malformed publish_time by trimming and safe parsing.
    """
    # Normalize columns
    df = df.toDF(*[c.strip() for c in df.columns])

    # Cast numeric columns with safe fallback (non-numeric -> 0)
    for c in ["views", "likes", "dislikes", "comment_count"]:
        if c in df.columns:
            df = df.withColumn(c, when(col(c).rlike("^[0-9]+$"), col(c)).otherwise("0").cast("long"))
        else:
            df = df.withColumn(c, expr("0").cast("long"))

    # category_id to int
    if "category_id" in df.columns:
        df = df.withColumn("category_id", when(col("category_id").rlike("^[0-9]+$"), col("category_id")).otherwise("0").cast("int"))
    else:
        df = df.withColumn("category_id", expr("0").cast("int"))

    # publish_time: slice to first 19 chars (YYYY-MM-DDTHH:MM:SS) and replace 'T' -> ' '
    if "publish_time" in df.columns:
        df = df.withColumn("publish_time_clean", expr("substring(publish_time, 1, 19)"))
        df = df.withColumn("publish_time_clean", expr("replace(publish_time_clean, 'T', ' ')"))
        # If Spark has try_to_timestamp we would use it; fallback to conditional to_timestamp
        df = df.withColumn("publish_time", when(col("publish_time_clean").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$"),
                                               to_timestamp(col("publish_time_clean"), "yyyy-MM-dd HH:mm:ss"))
                                     .otherwise(None))
        df = df.drop("publish_time_clean")
    else:
        df = df.withColumn("publish_time", expr("NULL").cast("timestamp"))

    # Fill boolean-like columns
    for c in ["comments_disabled", "ratings_disabled", "video_error_or_removed"]:
        if c in df.columns:
            df = df.withColumn(c, when(col(c).isNull(), "False").otherwise(col(c)))
        else:
            df = df.withColumn(c, expr("'False'"))

    return df

# ---------- Broadcast variables and accumulators ----------
def broadcast_and_accumulator_demo(spark, df):
    """
    Demonstrates broadcast variable (category mapping) and accumulator for bad rows count.
    """
    sc = spark.sparkContext
    print("\n--- Broadcast & Accumulator Demo ---")
    # broadcast category map
    cat_map = FALLBACK_CAT_MAP.copy()
    b_cat = sc.broadcast(cat_map)

    # old-style accumulator (compatible)
    bad_rows = sc.accumulator(0)

    # simple parsing example: check if title exists and is string; increment accumulator when bad
    def check_row(row):
        try:
            title = row['title']
            if not title or not isinstance(title, str):
                bad_rows.add(1)
            return 1
        except Exception:
            bad_rows.add(1)
            return 0

    # apply map on an RDD partition (no collect)
    _ = df.rdd.map(check_row).take(5)
    print("Accumulator (bad rows) value:", bad_rows.value)
    # illustrate use of broadcast variable: map category_id to name for few rows
    sample_cat_names = df.select("category_id").distinct().rdd.map(lambda r: (r[0], b_cat.value.get(r[0], "Other"))).take(10)
    print("Sample category_id -> name (via broadcast):", sample_cat_names[:10])
    return b_cat, bad_rows

# ---------- Partitioning, caching, checkpointing ----------
def partition_cache_checkpoint_demo(spark, df):
    sc = spark.sparkContext
    print("\n--- Partitioning, Caching, Checkpointing ---")
    print("Initial partitions (DataFrame.rdd):", df.rdd.getNumPartitions())
    # Repartition to 8 (demo)
    df_repart = df.repartition(8)
    print("After repartition:", df_repart.rdd.getNumPartitions())
    # Persist (cache)
    df_cached = df_repart.cache()
    # Force materialization
    _ = df_cached.count()
    # Checkpoint the RDD derived from DF
    rdd = df_cached.rdd
    rdd_checkpointable = rdd.map(lambda x: x)
    rdd_checkpointable.checkpoint()
    _ = rdd_checkpointable.count()
    print("Checkpointed RDD materialized.")
    return df_cached

# ---------- Sampling, external program, config printing, optimization tips ----------
def sampling_and_external_demo(spark, df):
    sc = spark.sparkContext
    print("\n--- Sampling & External Program ---")
    # Sample 1% (without replacement)
    sample_rdd = df.rdd.sample(False, 0.01, seed=42)
    print("Sampled rows (approx):", sample_rdd.count())
    # Call external program (echo) to show external integration
    result = subprocess.run(["echo", "External program invoked from PySpark project"], capture_output=True, text=True)
    print("External program output:", result.stdout.strip())

    # Print Spark config of interest
    print("Spark UI Web URL:", sc.uiWebUrl)
    print("Shuffle partitions:", spark.sparkContext.getConf().get("spark.sql.shuffle.partitions"))

    # Optimization tips demonstration (filter early)
    # Example: filter out videos with views < 1000 before heavy operations
    df_filtered = df.filter(col("views") >= 1000)
    return df_filtered

# ---------- Analytics: engagement, top categories, best hours, top videos ----------
def analytics_and_export(df, out_dir):
    print("\n--- Analytics & Exports ---")
    # Feature engineering
    df = df.withColumn("engagement", (col("likes") + col("comment_count")).cast("long"))
    df = df.withColumn("engagement_rate", when(col("views") > 0, col("engagement") / col("views")).otherwise(0.0))
    df = df.withColumn("publish_hour", hour(col("publish_time")))
    df = df.withColumn("publish_day", date_format(col("publish_time"), "E"))

    # Map category_id -> category_name via CASE (built safely)
    cases = "CASE "
    for cid, cname in FALLBACK_CAT_MAP.items():
        clean_name = cname.replace("'", "")  # remove single quotes to avoid SQL issues
        cases += f" WHEN category_id = {int(cid)} THEN '{clean_name}' "
    cases += " ELSE 'Other' END"
    df = df.withColumn("category_name", expr(cases))

    # Persist intermediate DF for repeated queries
    df = df.persist()

    # 1) Top categories by avg engagement_rate
    cat_agg = df.groupBy("category_name").agg(
        spark_avg("engagement_rate").alias("avg_engagement_rate"),
        spark_sum("views").alias("total_views"),
        spark_count("*").alias("num_records")
    ).orderBy(desc("avg_engagement_rate"))
    cat_out = os.path.join(out_dir, "top_categories.csv")
    cat_agg.coalesce(1).write.mode("overwrite").option("header", True).csv(cat_out)
    print("Wrote top categories to", cat_out)
    cat_agg.show(10, truncate=False)

    # 2) Best posting hours
    hour_agg = df.groupBy("publish_hour").agg(spark_avg("engagement_rate").alias("avg_rate")).orderBy(desc("avg_rate"))
    hour_out = os.path.join(out_dir, "best_hours.csv")
    hour_agg.coalesce(1).write.mode("overwrite").option("header", True).csv(hour_out)
    print("Wrote best hours to", hour_out)
    hour_agg.show(10, truncate=False)

    # 3) Top videos and channels
    top_videos = df.select("video_id", "title", "channel_title", "views", "likes", "comment_count", "engagement", "engagement_rate") \
                   .orderBy(desc("engagement")).limit(100)
    top_videos_out = os.path.join(out_dir, "top_videos.csv")
    top_videos.coalesce(1).write.mode("overwrite").option("header", True).csv(top_videos_out)
    print("Wrote top videos to", top_videos_out)
    top_videos.show(10, truncate=False)

    top_channels = df.groupBy("channel_title").agg(
        spark_sum("views").alias("total_views"),
        spark_sum("engagement").alias("total_engagement"),
        spark_avg("engagement_rate").alias("avg_engagement_rate"),
        spark_count("*").alias("num_videos")
    ).orderBy(desc("total_engagement")).limit(50)
    top_channels_out = os.path.join(out_dir, "top_channels.csv")
    top_channels.coalesce(1).write.mode("overwrite").option("header", True).csv(top_channels_out)
    print("Wrote top channels to", top_channels_out)
    top_channels.show(10, truncate=False)

    # 4) Trending keywords (simple overall top tokens from titles)
    tokens = df.select("title") \
               .withColumn("title_clean", lower(regexp_replace(col("title"), r"[^a-zA-Z0-9\s]", " "))) \
               .withColumn("token", explode(split(col("title_clean"), "\\s+"))) \
               .filter((col("token").isNotNull()) & (col("token") != "") & (col("token").rlike("^[a-zA-Z0-9]{2,}$")))
    token_counts = tokens.groupBy("token").agg(spark_count("*").alias("freq")).orderBy(desc("freq"))
    token_out = os.path.join(out_dir, "trending_keywords_overall.csv")
    token_counts.coalesce(1).write.mode("overwrite").option("header", True).csv(token_out)
    print("Wrote trending keywords to", token_out)
    token_counts.show(20, truncate=False)

    # Save cleaned parquet of dataset for reproducibility
    cleaned_parquet = os.path.join(out_dir, "cleaned.parquet")
    df.write.mode("overwrite").parquet(cleaned_parquet)
    print("Saved cleaned parquet to", cleaned_parquet)

    return df

# ----------  MLlib - ALS recommender (simple demo) ----------
def mllib_als_demo(spark):
    print("\n---MLlib ALS Recommender Demo ---")
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql.types import IntegerType, FloatType, StructType, StructField

    # Build tiny ratings DataFrame from top videos sample (toy example)
    sample_ratings = [
        (0, 10, 4.0), (0, 11, 2.0), (1, 10, 5.0), (1, 12, 3.0),
        (2, 11, 4.0), (2, 12, 1.0), (3, 10, 2.0), (3, 13, 5.0)
    ]
    schema = StructType([
        StructField("userId", IntegerType(), False),
        StructField("itemId", IntegerType(), False),
        StructField("rating", FloatType(), False)
    ])
    ratings = spark.createDataFrame(sample_ratings, schema=schema)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=42)
    als = ALS(maxIter=5, regParam=0.1, userCol="userId", itemCol="itemId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)
    preds = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(preds)
    print("ALS RMSE (toy):", rmse)
    print("Recommendations sample:")
    recs = model.recommendForAllUsers(2)
    recs.show(truncate=False)

# ---------- Main ----------
def main():
    spark = create_spark()
    sc = spark.sparkContext
    spark.sparkContext.setLogLevel("WARN")

    print("--Spark Session Started--")
    # 1) Read dataset
    df_raw = safe_read_csv(spark, DATASET_PATH)
    print("Initial rows (raw):", df_raw.count())
    df_raw.printSchema()

    # 2) Preprocess & clean
    df_clean = preprocess_df(df_raw)
    print("After cleaning sample:")
    df_clean.select("views", "likes", "comment_count", "category_id", "publish_time").show(5, truncate=False)

    # 3) UNIT-3 demos (RDD + pairRDD)
    wc = rdd_wordcount_demo(sc, df_clean)
    joined_pair, left_pair = pair_rdd_join_demo(sc)

    # 4) Broadcast & accumulator
    bcat, bad = broadcast_and_accumulator_demo(spark, df_clean)

    # 5) Partitioning / caching / checkpointing demo
    df_cached = partition_cache_checkpoint_demo(spark, df_clean)

    # 6) UNIT-4 sampling + external program + config print
    df_filtered = sampling_and_external_demo(spark, df_cached)

    # 7) Analytics & exports (top categories, hours, top videos, keywords)
    df_final = analytics_and_export(df_filtered, OUT_DIR)

    # 8) MLlib small demo
    mllib_als_demo(spark)

    # 9) Print completion and report file reference
    print("\nAll outputs saved to:", os.path.abspath(OUT_DIR))
    print("Report template file (uploaded) path:", REPORT_FILE)
    spark.stop()
    print("--Spark Stopped--")

if __name__ == "__main__":
    main()
