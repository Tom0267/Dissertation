#count the number of videos in each class in tested_video_ids.txt


import os

def count_videos_in_classes(tensors_folder):
    class_counts = {"non-violent": 0, "violent": 0, "unknown": 0}

    video_ids_file = "tested_video_ids.txt"  # File containing video IDs
    video_ids_path = os.path.join(tensors_folder, video_ids_file)
    if not os.path.exists(video_ids_path):
        print(f"File {video_ids_file} not found in {tensors_folder}.")
        return class_counts

    with open(video_ids_path, "r") as f:
        video_ids = [line.strip().lower() for line in f if line.strip()]

    for video_id in video_ids:
        if video_id.startswith("nonviolence") or video_id.startswith("nv"):
            class_counts["non-violent"] += 1
        elif video_id.startswith("violence") or video_id.startswith("v"):
            class_counts["violent"] += 1
        else:
            class_counts["unknown"] += 1

    return class_counts

if __name__ == "__main__":
    tensors_folder = "."  # Set this to the folder containing tested_video_ids.txt
    counts = count_videos_in_classes(tensors_folder)
    
    print("Number of videos in each class:")
    for class_name, count in counts.items():
        print(f"{class_name}: {count}")
    
    total_videos = sum(counts.values())
    print(f"Total videos processed: {total_videos}")