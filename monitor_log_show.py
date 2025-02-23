import pandas as pd
import matplotlib.pyplot as plt
import glob

# ✅ 가장 최근 저장된 CSV 파일 찾기
log_files = sorted(glob.glob("./training_logs/*.monitor.csv"), reverse=True)

if log_files:
    latest_log = log_files[0]  # 가장 최근 파일 선택
    print(f"📂 분석할 파일: {latest_log}")

    # ✅ 데이터 로드 및 시각화
    monitor_data = pd.read_csv(latest_log, comment="#")

    plt.plot(monitor_data.index, monitor_data["r"], label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("❌ 저장된 로그 파일이 없습니다.")