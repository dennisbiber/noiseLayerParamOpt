from safetensors.torch import save_file
import csv
import os
import yaml


def save_model_as_safetensor(model, filename):
    model_state_dict = model.state_dict()
    save_file(model_state_dict, filename)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def json2csv(conf, retunrGrid=False):
    keys = conf.keys()
    l1 = [(conf[k]["low"], conf[k]["high"]) for k in keys if k != "grid" and k != "invert"]
    grid = [(conf[k]["hY"]["low"], conf[k]["hY"]["high"], conf[k]["wX"]["low"], conf[k]["wX"]["high"]) for k in keys if k == "grid"]
    l = l1 + grid
    ks = [k for k in keys if k != "grid"]
    return l, ks

def writeRow(writer, row):
    writer.writerow(row)

def append_to_csv(conf, model_name, csv_file):
    l, ks = json2csv(conf)
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writeRow(writer, (ks + ["model_name"]))
            writeRow(writer, (l + [model_name]))
    else:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writeRow(writer, (l + [model_name]))