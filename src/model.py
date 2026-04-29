
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F


def create_spark_session(app_name="MovieRecommender"):
    """Khởi tạo Spark Session"""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


def load_data(spark, ratings_path, movies_path):
    """Load dữ liệu ratings và movies từ CSV"""
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
    movies  = spark.read.csv(movies_path,  header=True, inferSchema=True)
    return ratings, movies


def build_sparse_matrix(ratings):
    """
    FR1: Xây dựng Sparse Matrix
    Chỉ giữ 3 cột: userId, movieId, rating
    """
    return ratings.select("userId", "movieId", "rating")


def train_als_model(ratings_df, rank=10, maxIter=10, regParam=0.1):
    """
    FR2: Train ALS model để học Latent Factors
    Trả về: model đã train + RMSE
    """
    # Chia train/test
    train, test = ratings_df.randomSplit([0.8, 0.2], seed=42)

    # Khởi tạo ALS
    als = ALS(
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )

    # Train
    print("⏳ Đang train ALS model...")
    model = als.fit(train)

    # Đánh giá RMSE
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"✅ Train xong! RMSE = {rmse:.4f}")

    return model, rmse


def get_top10_recommendations(model, user_id, movies_df, ratings_df):
    """
    Output: Nhận UserID → trả về Top 10 phim gợi ý
    kèm điểm dự đoán
    """
    spark = SparkSession.getActiveSession()

    # Phim user chưa xem
    watched   = ratings_df.filter(F.col("userId") == user_id).select("movieId")
    unwatched = movies_df.select("movieId").subtract(watched)

    # Tạo cặp (userId, movieId)
    user_df     = spark.createDataFrame([(user_id,)], ["userId"])
    user_movies = unwatched.crossJoin(user_df)

    # Dự đoán + lấy Top 10
    top10 = model.transform(user_movies) \
                 .orderBy(F.col("prediction").desc()) \
                 .limit(10) \
                 .join(movies_df, "movieId") \
                 .select("movieId", "title", "genres", "prediction")

    return top10


def handle_cold_start(movies_df, ratings_df, n=10):
    """
    FR3: Xử lý Cold Start
    User mới chưa có data → gợi ý n phim phổ biến nhất
    """
    popular = ratings_df \
        .groupBy("movieId") \
        .agg(
            F.count("rating").alias("num_ratings"),
            F.avg("rating").alias("avg_rating")
        ) \
        .filter(F.col("num_ratings") >= 100) \
        .orderBy(F.col("avg_rating").desc()) \
        .limit(n) \
        .join(movies_df, "movieId") \
        .select("movieId", "title", "genres", "avg_rating", "num_ratings")

    return popular


def save_model(model, path):
    """Lưu model ra disk"""
    model.save(path)
    print(f"✅ Đã lưu model vào {path}")


def load_model(path):
    """Load model đã lưu"""
    from pyspark.ml.recommendation import ALSModel
    model = ALSModel.load(path)
    print(f"✅ Đã load model từ {path}")
    return model
