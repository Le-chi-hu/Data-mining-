# Data Mining 資料探勘

  長庚大學資料探勘課程的作業與期末專案紀錄，主要參考教科書《Learning Data Mining with Python》。

  ## 目錄結構

  ```
  Data-mining-/
  ├── HW1/                # 關聯規則探勘
  ├── HW2/                # k-近鄰演算法
  ├── HW3/                # 決策樹與隨機森林
  ├── HW4/                # 文字分群
  ├── Final project/      # 期末專案：交通事故風險預測
  └── README.md
  ```

  ## 作業內容

  | 作業 | 主題 | 資料集 | 使用技術 |
  |------|------|--------|----------|
  | HW1 | 關聯規則探勘（Apriori） | MovieLens 100K | Pandas、SciPy、Lift、Chi² |
  | HW2 | k-NN 分類與樣本濃縮 | Ionosphere Dataset | Scikit-learn、MinMaxScaler、Pipeline、CNN 演算法 |
  | HW3 | NBA 比賽結果預測 | NBA 2013–2014 | 決策樹、隨機森林、GridSearchCV |
  | HW4 | 新聞標題文字分群 | UCI 新聞聚合資料集 | TF-IDF、KMeans、EAC、Silhouette Score |

  ## 期末專案

  **交通事故風險預測**（Kaggle Playground Series S5E10）

  - **目標**：根據道路與環境特徵預測交通事故風險程度
  - **資料量**：訓練集 517,754 筆、測試集 172,585 筆
  - **特徵**：道路類型、車道數、速度限制、天氣、照明、時段、假日等
  - **工具**：Pandas、Matplotlib

  ## 環境需求

  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  scipy
  jupyter
  ```

  ## 參考教材

  - *Learning Data Mining with Python* — Robert Layton

  ---

  # Data Mining

  Course assignments and final project from the Data Mining class at Chang Gung University.
  Main reference: *Learning Data Mining with Python* by Robert Layton.

  ## Repository Structure

  ```
  Data-mining-/
  ├── HW1/                # Association Rule Mining
  ├── HW2/                # k-Nearest Neighbors
  ├── HW3/                # Decision Tree & Random Forest
  ├── HW4/                # Text Clustering
  ├── Final project/      # Road Accident Risk Prediction
  └── README.md
  ```

  ## Assignments

  | Assignment | Topic | Dataset | Technologies |
  |------------|-------|---------|--------------|
  | HW1 | Association Rule Mining (Apriori) | MovieLens 100K | Pandas、SciPy、Lift、Chi² |
  | HW2 | k-NN Classification & Sample Condensing | Ionosphere Dataset | Scikit-learn、MinMaxScaler、Pipeline、CNN
  Algorithm |
  | HW3 | NBA Game Result Prediction | NBA 2013–2014 | Decision Tree、Random Forest、GridSearchCV |
  | HW4 | News Headline Text Clustering | UCI News Aggregator | TF-IDF、KMeans、EAC、Silhouette Score |

  ## Final Project

  **Predicting Road Accident Risk** (Kaggle Playground Series S5E10)

  - **Goal**: Predict traffic accident risk levels based on road and environmental features
  - **Data**: 517,754 training samples / 172,585 test samples
  - **Features**: Road type, lane count, speed limit, weather, lighting, time of day, holidays, etc.
  - **Tools**: Pandas、Matplotlib

  ## Requirements

  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  scipy
  jupyter
  ```

  ---
