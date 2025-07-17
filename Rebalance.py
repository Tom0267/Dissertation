import pandas as pd

results_csv = "RWF-2000/DeepseekResults.csv"
output_file = "tested_video_ids.txt"

df = pd.read_csv(results_csv)

df["VideoID"] = df["VideoID"].astype(str).str.strip().str.lower()

def inferLabel(video_id):
    if video_id == "nan":
        print(video_id)
    #video_id = video_id.lower()
    if video_id.startswith("violence"):
        return "violent"
    elif video_id.startswith("nonviolence"):
        return "non-violent"
    return "unknown"

#apply the label inference
df["TrueLabel"] = df["VideoID"].apply(inferLabel)

#check for duplicate video IDs and print both IDs and their labels
duplicates = df[df.duplicated(subset=["VideoID"], keep=False)]
if not duplicates.empty:
    print("Duplicate Video IDs found:")
    for index, row in duplicates.iterrows():
        print(f"VideoID: {row['VideoID']}, TrueLabel: {row['TrueLabel']}")
        
#remove duplicates by keeping the first occurrence
df = df.drop_duplicates(subset=["VideoID"], keep='first')

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