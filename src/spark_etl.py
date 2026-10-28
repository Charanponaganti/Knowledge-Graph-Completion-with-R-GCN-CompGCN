"""PySpark ETL for knowledge-graph triples.

Mirrored in the Kaggle notebook. Not imported by the notebook.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession, Window, functions as F
from pyspark.sql.types import StringType, StructField, StructType


TRIPLE_SCHEMA = StructType(
    [
        StructField("h", StringType(), False),
        StructField("r", StringType(), False),
        StructField("t", StringType(), False),
    ]
)


def build_spark(app_name: str = "kg-completion", driver_memory: str = "12g",
                shuffle_partitions: int = 8, local_dir: str = "/tmp/spark-tmp") -> SparkSession:
    """Kaggle-friendly local Spark session. Stop it before GPU training."""
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "6g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.local.dir", local_dir)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def load_triples(spark: SparkSession, path: str, delimiter: str = "\t") -> DataFrame:
    return (
        spark.read.option("header", "false")
        .option("delimiter", delimiter)
        .schema(TRIPLE_SCHEMA)
        .csv(path)
    )


def clean_triples(df: DataFrame, drop_self_loops: bool = True) -> DataFrame:
    out = df.dropna().dropDuplicates(["h", "r", "t"])
    out = out.filter((F.col("h") != "") & (F.col("r") != "") & (F.col("t") != ""))
    if drop_self_loops:
        out = out.filter(F.col("h") != F.col("t"))
    return out


def entity_degree(df: DataFrame) -> DataFrame:
    heads = df.select(F.col("h").alias("e"))
    tails = df.select(F.col("t").alias("e"))
    return heads.union(tails).groupBy("e").count().withColumnRenamed("count", "degree")


def relation_frequency(df: DataFrame) -> DataFrame:
    return df.groupBy("r").count().withColumnRenamed("count", "freq").orderBy(F.desc("freq"))


def relation_categories(df: DataFrame, threshold: float = 1.5) -> DataFrame:
    """Bordes-style 1-1 / 1-N / N-1 / N-N labels using tph and hpt."""
    tph = (
        df.groupBy("r", "h")
        .agg(F.countDistinct("t").alias("n_t"))
        .groupBy("r")
        .agg(F.avg("n_t").alias("tph"))
    )
    hpt = (
        df.groupBy("r", "t")
        .agg(F.countDistinct("h").alias("n_h"))
        .groupBy("r")
        .agg(F.avg("n_h").alias("hpt"))
    )
    joined = tph.join(hpt, "r")
    return joined.withColumn(
        "category",
        F.when((F.col("tph") < threshold) & (F.col("hpt") < threshold), "1-1")
        .when((F.col("tph") >= threshold) & (F.col("hpt") < threshold), "1-N")
        .when((F.col("tph") < threshold) & (F.col("hpt") >= threshold), "N-1")
        .otherwise("N-N"),
    )


def graph_stats(df: DataFrame) -> Dict[str, int]:
    n_ent = df.select("h").union(df.select("t")).distinct().count()
    n_rel = df.select("r").distinct().count()
    n_tri = df.count()
    return {"n_entities": n_ent, "n_relations": n_rel, "n_triples": n_tri}


def degree_filter(df: DataFrame, min_degree: int, top_k: Optional[int] = None) -> DataFrame:
    deg = entity_degree(df)
    keep = deg.filter(F.col("degree") >= min_degree)
    if top_k is not None:
        keep = keep.orderBy(F.desc("degree")).limit(int(top_k))
    keep_ids = keep.select(F.col("e").alias("id"))
    return (
        df.join(keep_ids.withColumnRenamed("id", "h"), "h", "inner")
        .join(keep_ids.withColumnRenamed("id", "t"), "t", "inner")
        .select("h", "r", "t")
    )


def k_core(df: DataFrame, k: int, max_iters: int = 20) -> DataFrame:
    """Iterative k-core on an undirected projection of the KG."""
    current = df
    for _ in range(max_iters):
        deg = entity_degree(current).filter(F.col("degree") >= k)
        nxt = (
            current.join(deg.select(F.col("e").alias("h")), "h", "inner")
            .join(deg.select(F.col("e").alias("t")), "t", "inner")
            .select("h", "r", "t")
        )
        if nxt.count() == current.count():
            break
        current = nxt
    return current


def dense_index(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Map string entities/relations to 0-based dense ids via window rank."""
    ents = df.select(F.col("h").alias("name")).union(df.select(F.col("t").alias("name"))).distinct()
    rels = df.select(F.col("r").alias("name")).distinct()
    e_w = Window.orderBy("name")
    r_w = Window.orderBy("name")
    e_map = ents.withColumn("id", F.row_number().over(e_w) - 1)
    r_map = rels.withColumn("id", F.row_number().over(r_w) - 1)
    indexed = (
        df.join(e_map.withColumnRenamed("name", "h").withColumnRenamed("id", "h_id"), "h")
        .join(r_map.withColumnRenamed("name", "r").withColumnRenamed("id", "r_id"), "r")
        .join(e_map.withColumnRenamed("name", "t").withColumnRenamed("id", "t_id"), "t")
        .select(F.col("h_id").alias("h"), F.col("r_id").alias("r"), F.col("t_id").alias("t"))
    )
    return indexed, e_map, r_map


def add_inverse_relations(df: DataFrame, n_relations: Optional[int] = None) -> DataFrame:
    """Add (t, r+offset, h) reciprocal triples. Offset defaults to max(r)+1."""
    if n_relations is None:
        n_relations = df.agg(F.max("r")).collect()[0][0] + 1
    inv = df.select(
        F.col("t").alias("h"),
        (F.col("r") + F.lit(int(n_relations))).alias("r"),
        F.col("h").alias("t"),
    )
    return df.unionByName(inv)


def filter_unseen_entities(train: DataFrame, split: DataFrame) -> DataFrame:
    seen = train.select(F.col("h").alias("e")).union(train.select(F.col("t").alias("e"))).distinct()
    return (
        split.join(seen.withColumnRenamed("e", "h"), "h", "inner")
        .join(seen.withColumnRenamed("e", "t"), "t", "inner")
        .select("h", "r", "t")
    )


def filtered_lookups(all_true: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """true tails per (h,r) and true heads per (t,r) for filtered ranking."""
    true_tails = all_true.groupBy("h", "r").agg(F.collect_set("t").alias("true_set"))
    true_heads = all_true.groupBy("t", "r").agg(F.collect_set("h").alias("true_set"))
    return true_tails, true_heads


def write_parquet(df: DataFrame, path: str, n_partitions: int = 4) -> None:
    df.coalesce(n_partitions).write.mode("overwrite").parquet(path)
