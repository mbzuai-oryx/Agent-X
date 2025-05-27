# === Dynamic nth frame based on video length ===
def decide_nth_frame(total_frames):
    if total_frames <= 50:
        return 3
    elif total_frames <= 200:
        return 10
    elif total_frames <= 500:
        return 20
    elif total_frames <= 1000:
        return 30
    else:
        return 50

# === Frame extraction ===
def extract_every_nth_frame_dynamic(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

    if not video_files:
        print("⚠️ No video files found.")
        return

    for video_file in sorted(video_files):
        video_path = os.path.join(video_folder, video_file)
        basename = os.path.splitext(video_file)[0]

        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        nth_frame = decide_nth_frame(total_frames)

        print(f"🎞 Extracting every {nth_frame}th frame from {video_file} ({total_frames} frames total)...")

        count = 0
        saved = 0
        success, image = vidcap.read()

        while success:
            if count % nth_frame == 0:
                frame_filename = f"{basename}_frame{saved}.jpg"
                save_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(save_path, image)
                saved += 1
                print(f"✅ Saved frame {saved} at {save_path}")
            success, image = vidcap.read()
            count += 1

        vidcap.release()

    print("\n🚀 Finished extracting frames dynamically.")
