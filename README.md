# 🎬 Movie Recommender System

Hệ thống gợi ý phim sử dụng **PySpark ALS** trên MovieLens 20M.

---

## 📁 Cấu trúc dự án

```
DO_AN_BIGDATA/
├── data/               # Dữ liệu MovieLens (KHÔNG push lên Git)
├── docs/               # Báo cáo, biểu đồ EDA
├── notebooks/          # Các notebook Jupyter
├── src/                # Mã nguồn chính
│   ├── __init__.py
│   └── model.py
├── tests_nghiep_vu/    # Unit Tests
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Cài đặt & Chạy

### 1. Cài thư viện
```bash
pip install -r requirements.txt
```

### 2. Tải dữ liệu
Chạy notebook `notebooks/01_download_data.ipynb`

### 3. EDA
Chạy notebook `notebooks/02_eda.ipynb`

### 4. Train model
Chạy notebook `notebooks/03_train_model.ipynb`

### 5. Chạy Unit Test
```bash
pytest tests_nghiep_vu/ -v
```

---

## 📊 Dữ liệu

| Thông tin | Chi tiết |
|-----------|----------|
| Dataset | MovieLens 20M |
| Số ratings | 20,000,263 |
| Số users | 138,493 |
| Số movies | 27,278 |

---

## ⚙️ Mô hình ALS

| Tham số | Giá trị |
|---------|---------|
| rank | 10 |
| maxIter | 10 |
| regParam | 0.1 |

---

## 🔧 Chức năng chính

**FR1 - Sparse Matrix:** Xây dựng ma trận thưa User x Item.

**FR2 - ALS Model:** Học Latent Factors của user và phim.

**FR3 - Cold Start:** User mới → gợi ý phim phổ biến nhất.

**Output:** Nhận UserID → trả về Top 10 phim kèm điểm dự đoán.

---

## 📈 Kết quả

- RMSE trên tập test: ~0.80

---

## 🛠️ Công nghệ sử dụng

- Python 3.10+
- PySpark 3.5.0 / Spark MLlib
- Google Colab + GitHub
