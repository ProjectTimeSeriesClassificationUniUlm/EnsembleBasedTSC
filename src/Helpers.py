from csv import DictWriter
import csv

def append_to_csv(csv_path, input_row):
    with open(csv_path, 'a', newline='') as f_object:
        writer = csv.writer(f_object)
        writer.writerow(input_row)

