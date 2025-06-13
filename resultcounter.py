import csv
def countResults(file_path):
    violent_count = 0
    nonviolent_count = 0

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['VideoID'].startswith('Violence'):
                violent_count += 1
            elif row['VideoID'].startswith('NonViolence'):
                nonviolent_count += 1

    return violent_count, nonviolent_count
if __name__ == "__main__":
    file_path = 'RWF-2000/results.csv'
    violent_count, nonviolent_count = countResults(file_path)
    print(f"Violent results: {violent_count}")
    print(f"Nonviolent results: {nonviolent_count}")