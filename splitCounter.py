import json

def countSamples(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    violent_count = sum(1 for key in data if key.startswith('Violence'))
    non_violent_count = sum(1 for key in data if key.startswith('NonViolence'))

    return violent_count, non_violent_count

if __name__ == "__main__":
    file_path = 'RWF-2000/motion_features.json'
    violent_count, non_violent_count = countSamples(file_path)
    print(f"Violent samples: {violent_count}")
    print(f"Non-violent samples: {non_violent_count}")
    print(f"Total samples: {violent_count + non_violent_count}")
