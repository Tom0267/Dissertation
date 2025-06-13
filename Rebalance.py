import pandas as pd

results_csv = "RWF-2000/results.csv"
output_file = "tested_video_ids.txt"

df = pd.read_csv(results_csv)

def inferLabel(video_id):
    video_id = video_id.lower()
    if video_id.startswith("violence"):
        return "violent"
    elif video_id.startswith("nonviolence"):
        return "non-violent"
    return "unknown"

#apply the label inference
df["TrueLabel"] = df["VideoID"].apply(inferLabel)

#keep only recognised labels
df = df[df["TrueLabel"] != "unknown"]

#extract and save the list of successfully tested video IDs
successful_ids = df["VideoID"].tolist()
with open(output_file, "w") as f:
    for vid in successful_ids:
        f.write(f"{vid}\n")

#count class distribution
class_counts = df["TrueLabel"].value_counts().to_dict()

print(f"Saved {len(successful_ids)} successfully tested video IDs to {output_file}")
print("\nNumber of videos processed by true class:")
print(class_counts)