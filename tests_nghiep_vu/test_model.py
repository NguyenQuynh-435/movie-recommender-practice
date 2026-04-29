
import pytest
import sys
sys.path.insert(0, ".")

from pyspark.sql import SparkSession
from src.model import build_sparse_matrix, handle_cold_start


@pytest.fixture(scope="session")
def spark():
    """Tạo Spark Session dùng chung cho tất cả test"""
    return SparkSession.builder \
        .appName("TestMovieRecommender") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()


# ─── TEST FR1: Sparse Matrix ───────────────────────────

def test_sparse_matrix_co_3_cot(spark):
    """FR1: Kết quả phải có đúng 3 cột"""
    data = [(1, 101, 4.0), (2, 102, 3.5), (3, 103, 5.0)]
    df = spark.createDataFrame(data, ["userId", "movieId", "rating"])
    
    result = build_sparse_matrix(df)
    
    assert set(result.columns) == {"userId", "movieId", "rating"}
    print("✅ test_sparse_matrix_co_3_cot PASSED")


def test_sparse_matrix_khong_null(spark):
    """FR1: Không được có giá trị null trong rating"""
    data = [(1, 101, 4.0), (2, 102, 3.5)]
    df = spark.createDataFrame(data, ["userId", "movieId", "rating"])
    
    result = build_sparse_matrix(df)
    null_count = result.filter("rating IS NULL").count()
    
    assert null_count == 0
    print("✅ test_sparse_matrix_khong_null PASSED")


def test_sparse_matrix_so_dong_khong_doi(spark):
    """FR1: Số dòng không được thay đổi sau khi tạo sparse matrix"""
    data = [(1, 101, 4.0), (2, 102, 3.5), (3, 103, 5.0)]
    df = spark.createDataFrame(data, ["userId", "movieId", "rating"])
    
    result = build_sparse_matrix(df)
    
    assert result.count() == 3
    print("✅ test_sparse_matrix_so_dong_khong_doi PASSED")


# ─── TEST FR3: Cold Start ───────────────────────────────

def test_cold_start_tra_ve_dung_so_phim(spark):
    """FR3: Phải trả về đúng n phim yêu cầu"""
    ratings_data = [(i, j, 4.0) for i in range(1, 200) for j in range(1, 6)]
    movies_data  = [(j, f"Movie {j}", "Action") for j in range(1, 6)]
    
    ratings_df = spark.createDataFrame(ratings_data, ["userId","movieId","rating"])
    movies_df  = spark.createDataFrame(movies_data,  ["movieId","title","genres"])
    
    result = handle_cold_start(movies_df, ratings_df, n=3)
    
    assert result.count() == 3
    print("✅ test_cold_start_tra_ve_dung_so_phim PASSED")


def test_cold_start_co_du_cot(spark):
    """FR3: Kết quả phải có đủ các cột cần thiết"""
    ratings_data = [(i, j, 4.0) for i in range(1, 200) for j in range(1, 6)]
    movies_data  = [(j, f"Movie {j}", "Action") for j in range(1, 6)]
    
    ratings_df = spark.createDataFrame(ratings_data, ["userId","movieId","rating"])
    movies_df  = spark.createDataFrame(movies_data,  ["movieId","title","genres"])
    
    result = handle_cold_start(movies_df, ratings_df, n=3)
    required_cols = {"movieId", "title", "genres", "avg_rating", "num_ratings"}
    
    assert required_cols.issubset(set(result.columns))
    print("✅ test_cold_start_co_du_cot PASSED")
