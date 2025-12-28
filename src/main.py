import cv2
import numpy as np
import math
import argparse
import os


class VehicleSpeedEstimator:
    def __init__(self, video_path, output_name="output.mp4"):
        # ========== 1. 初始化與基本設定 ==========
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到影片檔案: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dt = 1.0 / self.fps

        # 參數設定 (完全依照原本限制條件)
        self.N = 5  # 移動平均長度
        self.speed_history = []
        self.prev_ref_segments = []
        self.frame_count = 0
        self.smooth_speed = None

        # 輸出設定
        self.out = cv2.VideoWriter(
            output_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height)
        )

        self.binary_width = 600
        self.binary_height = 300
        self.out_binary = cv2.VideoWriter(
            "output_binary.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.binary_width, self.binary_height)
        )

        # ========== 2. 透視變換設定 (保留原本座標) ==========
        self.src_pts = np.float32([
            [1457, 1129],  # 左上
            [1700, 1126],  # 右上
            [1930, 1295],  # 右下
            [1165, 1286]  # 左下
        ])

        # 計算變換矩陣 M
        self.width = int(np.linalg.norm(self.src_pts[0] - self.src_pts[1]))
        self.height = int(np.linalg.norm(self.src_pts[0] - self.src_pts[3]))

        dst_pts = np.float32([
            [0, 0],
            [self.width - 1, 0],
            [self.width - 1, self.height - 1],
            [0, self.height - 1]
        ])
        self.M = cv2.getPerspectiveTransform(self.src_pts, dst_pts)

    def _preprocess(self, frame):
        """步驟 1-3: 透視變換 + 二值化 + 形態學"""
        # Step 1: 鳥瞰轉換
        frame_bird = cv2.warpPerspective(frame, self.M, (self.width, self.height))

        # Step 2: 灰階 + 降噪 + 二值化 (閾值 130)
        gray = cv2.cvtColor(frame_bird, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)

        # Step 3: 形態學操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        return frame_bird, binary_clean

    def _get_segments(self, binary_clean):
        """步驟 4: 連通元件分析與篩選"""
        h_img, w_img = binary_clean.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)

        ref_segments = []
        for i in range(1, num_labels):
            x, y, w_box, h_box, area = stats[i]
            cx, cy = centroids[i]

            # 原本的篩選條件
            if w_box < (w_img * 2 / 5):
                if y + h_box / 2 > (h_img / 2):
                    ref_segments.append((x, y, w_box, h_box, cx, cy))
        return ref_segments

    def _calculate_scale_from_lines(self, binary_clean, frame_bird):
        """步驟 5: 霍夫直線變換取得比例尺"""
        scale_m_per_pixel = None
        edges = cv2.Canny(binary_clean, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=20)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                angle = abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))

                # 原本的角度過濾條件 (80 < angle < 100)
                if 80 < angle < 100:
                    cv2.line(frame_bird, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if length > 0:
                        scale_m_per_pixel = 10.0 / length
                    break
        return scale_m_per_pixel

    def _compute_speed(self, ref_segments, scale_m_per_pixel):
        """步驟 6: 計算速度 (每 2 幀一次)"""
        if self.frame_count % 2 == 0:
            for seg in ref_segments:
                x, y, w_box, h_box, cx, cy = seg

                # Fallback scale (依照原本邏輯)
                current_scale = scale_m_per_pixel
                if current_scale is None and h_box > 0:
                    current_scale = 10.0 / h_box

                best = None
                best_dist = 1e9
                for prev in self.prev_ref_segments:
                    pcx, pcy = prev[4], prev[5]
                    d = math.hypot(cx - pcx, cy - pcy)
                    if d < best_dist:
                        best_dist = d
                        best = prev

                # 原本的判斷條件 (dist < 50)
                if best is not None and best_dist < 50 and current_scale is not None:
                    real_dist = best_dist * current_scale

                    # 速度公式: 距離 / 時間 (dt * 2) * 3.6
                    speed = real_dist / (self.dt * 2) * 3.6

                    # 原本的過濾範圍 (80 ~ 120)
                    if 80 <= speed <= 120:
                        self.speed_history.append(speed)
                        if len(self.speed_history) > self.N:
                            self.speed_history.pop(0)

                        self.smooth_speed = sum(self.speed_history) / len(self.speed_history)
                        print(f"speed (smoothed) = {int(self.smooth_speed)}")

        # 更新上一幀的紀錄
        self.prev_ref_segments = ref_segments

    def _draw_results(self, frame, frame_bird, ref_segments):
        """繪製結果與輸出"""
        # 標註速度
        if self.smooth_speed is not None:
            text = f"Speed: {self.smooth_speed:.2f} km/h"
            cv2.putText(frame, text, (self.frame_width - 450, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 標註紅點 (鳥瞰圖)
        points_frame = frame_bird.copy()
        for i, seg in enumerate(ref_segments):
            cx, cy = int(seg[4]), int(seg[5])
            cv2.circle(points_frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(points_frame, str(i), (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 顯示畫面
        frame_out = cv2.resize(frame, (800, 500))
        cv2.imshow("Output", frame_out)
        cv2.imshow("Bird View Points", points_frame)

        return points_frame

    def run(self):
        print(f"開始處理影片...")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            # 1. 影像處理
            frame_bird, binary_clean = self._preprocess(frame)

            # 2. 取得連通元件
            ref_segments = self._get_segments(binary_clean)

            # 3. 取得比例尺 (每幀重新計算，依照原始邏輯)
            scale_m_per_pixel = self._calculate_scale_from_lines(binary_clean, frame_bird)

            # 4. 計算速度
            self._compute_speed(ref_segments, scale_m_per_pixel)

            # 5. 繪製與輸出
            self._draw_results(frame, frame_bird, ref_segments)

            # 寫入影片檔
            self.out.write(frame)

            binary_bgr = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)
            binary_resized = cv2.resize(binary_bgr, (self.binary_width, self.binary_height))
            self.out_binary.write(binary_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 釋放資源
        self.cap.release()
        self.out.release()
        self.out_binary.release()
        cv2.destroyAllWindows()
        print("處理完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/test1_mute.mp4")
    args = parser.parse_args()

    estimator = VehicleSpeedEstimator(video_path=args.video)
    estimator.run()