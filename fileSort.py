import os
import shutil

# Paths
video_id_file = "tested_video_ids.txt"
source_folder = "RWF-2000/extractedFrames"
destination_folder = "TestedFrames"


# Ensure destination exists
os.makedirs(destination_folder, exist_ok=True)

# Build a set of lowercase video prefixes from the txt file
video_prefixes = set()

with open(video_id_file, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.strip().split('_')
            if len(parts) >= 3:
                label = parts[0].capitalize()  # Matches "Violence" vs "NonViolence"
                vid_id = parts[1]
                index = parts[2]
                prefix = f"{label}_{vid_id}_{index}_".lower()
                video_prefixes.add(prefix)

# Count copied videos (not just files)
copied_video_keys = set()
copied_image_count = 0

# Scan source folder
for filename in os.listdir(source_folder):
    filename_lower = filename.lower()
    for prefix in video_prefixes:
        if filename_lower.startswith(prefix):
            src = os.path.join(source_folder, filename)
            dst = os.path.join(destination_folder, filename)
            shutil.copy2(src, dst)
            copied_video_keys.add(prefix)
            copied_image_count += 1
            print(f"Copied: {filename}")
            break

# Final report
print(f"\nTotal unique videos matched: {len(copied_video_keys)}")
print(f"Total images copied: {copied_image_count}")