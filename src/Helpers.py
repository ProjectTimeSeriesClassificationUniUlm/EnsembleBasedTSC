from csv import DictWriter


def append_to_csv(csv_path, column_names, input_row):
    input_dict = dict(zip(column_names, input_row))
    with open(csv_path, 'a') as f_object:
        dict_writer_object = DictWriter(f_object, fieldnames=column_names)
        dict_writer_object.writerow(input_dict)
        f_object.close()

