# Vehicle-Speed-Estimation-CV
# 基於逆透視轉換 (IPM) 之單目視覺車速估測系統

## 📖 專案簡介 (Project Overview)

本專案實作了一套即時的**單目視覺車速估測系統 (Vehicle Speed Estimation System)**。不同於傳統依賴雷達或光達 (Lidar) 的測速方案，本系統利用**電腦視覺 (Computer Vision)** 技術，直接從一般的 2D 監控影片中計算車輛速度。

核心技術採用 **逆透視變換 (Inverse Perspective Mapping, IPM)**，將攝影機的透視視角轉換為**鳥瞰圖 (Bird's-Eye View, BEV)**，藉此消除透視變形，實現精確的物理距離測量與速度計算。

### 🎯 核心功能 (Key Features)
* **逆透視變換 (IPM):** 利用透視變換矩陣，將 2D 影像還原為鳥瞰視角，建立線性距離關係。
* **動態比例尺校正 (Dynamic Scale Calibration):** 使用 **霍夫變換 (Hough Transform)** 自動偵測道路標線（虛線），動態計算「像素/公尺」的比例尺，適應不同路段。
* **強健的物件追蹤 (Robust Object Tracking):** 結合連通元件分析 (Connected Components) 與最近鄰搜索算法，穩定追蹤車輛軌跡。
* **訊號平滑化 (Signal Smoothing):** 實作 **指數移動平均 (EMA)** 濾波器，有效去除偵測雜訊，讓速度數值更穩定。
* **物件導向設計 (OOP):** 採用模組化的 `VehicleSpeedEstimator` 類別設計，程式碼結構清晰，易於維護與擴充。

## 🛠️ 安裝與環境 (Installation)

1.  **複製專案 (Clone)**
    ```bash
    git clone [https://github.com/sazxc77560/Vehicle-Speed-Estimation.git](https://github.com/sazxc77560/Vehicle-Speed-Estimation.git)
    cd Vehicle-Speed-Estimation
    ```

2.  **安裝依賴套件 (Install dependencies)**
    建議使用虛擬環境 (Virtual Environment) 執行。
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 使用方法 (Usage)

請在專案根目錄執行主程式。你可以透過指令參數指定輸入影片與輸出路徑。

```bash
# 基本用法 (使用預設設定)
python src/main.py --video data/test1_mute.mp4

# 指定輸出檔名
python src/main.py --video data/test1_mute.mp4 --output output/result_v1.mp4
```

## ⚙️ 參數校正與設定 (Calibration)

車速估算的準確度高度依賴於 **透視變換 (Perspective Transform)** 的參數。由於每支影片的攝影機架設角度不同，更換測試影片時，你必須手動校正 **感興趣區域 (ROI)** 的四個座標點。

請打開 `src/main.py` 並找到 `ROI_POINTS` 設定區塊：

```python
# 設定路面的四個角落座標 [左上, 右上, 右下, 左下]
ROI_POINTS_TEST1 = [
    [1457, 1129],
    [1700, 1126],
    [1930, 1295],
    [1165, 1286]
]
```
* **`real_world_length`**: 系統預設道路標線（虛線）的實際長度為 **10.0 公尺**。若測試場景的道路規範不同，可在 `VehicleSpeedEstimator` 初始化時調整此參數。
