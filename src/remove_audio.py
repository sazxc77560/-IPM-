import subprocess
import os

def remove_audio(input_file, output_file):
    """
    使用 FFmpeg 移除音訊，並且使用 'copy' 模式確保畫質無損。
    """
    if not os.path.exists(input_file):
        print(f"找不到檔案: {input_file}")
        return

    # 構建 FFmpeg 指令
    # -i: 輸入檔案
    # -c:v copy: 影像編碼 (Video Codec) 直接複製，不重新編碼 (關鍵!)
    # -an: Audio None (移除音訊)
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'copy',
        '-an',
        output_file,
        '-y'  # 如果檔案存在則直接覆蓋
    ]

    try:
        print(f"正在處理: {input_file} ...")
        subprocess.run(command, check=True)
        print(f"成功！已輸出至: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"發生錯誤: {e}")
    except FileNotFoundError:
        print("錯誤: 找不到 FFmpeg。請確認已安裝 FFmpeg 並加入系統路徑。")

# 使用範例
if __name__ == "__main__":
    remove_audio("data/test1.mp4", "data/test1_mute.mp4")