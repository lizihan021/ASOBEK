import modelclasses as model
import csv
import re

TRAIN_LABEL_REGEX = re.compile("\((\d),\s*(\d)\)\s?")

class InvalidLabelFormat(Exception):
    pass

def _train_label_from_str(str):
    if str == "0\n" or str == "0" :
        str = "(0, 5)"
    elif str == "1\n" or str == "1" :
        str = "(5, 0)"
    else :
        print(str)
        raise InvalidLabelFormat("label \"\" is in incorrect format")
    match = TRAIN_LABEL_REGEX.fullmatch(str)
    if match is None:
        raise InvalidLabelFormat("label \"\" is in incorrect format")
    return tuple(map(int, match.group(1, 2)))

TrainLabel = model.VarType(
    from_str = _train_label_from_str
)

class TrainDataRow(model.Model):
    topic_id = model.Integer
    s1_id = model.Integer
    s2_id = model.Integer
    sent_1 = model.String
    sent_2 = model.String
    label = TrainLabel

    def is_paraphrase(self):
        return self.label[0] >= 3

    def is_debatable(self):
        return self.label[0] is 2

class TestDataRow(model.Model):
    topic_id = model.Integer
    sent_1 = model.String
    sent_2 = model.String
    # label = model.Integer

    # def is_paraphrase(self):
    #     return self.label in (4, 5)

    # def is_debatable(self):
    #     return self.label is 3

# def read_database(filename, datarow_class):
#     with open(filename) as io:
#         # for line in io:
#         #     print(line.split("\t"))
#         return [datarow_class.from_tuple(line.split("\t")) for line in io]

def read_database(filename, datarow_class):
    with open(filename) as io:
        csv.field_size_limit(700000)
        csvreader = csv.reader(io)
        return [datarow_class.from_tuple(line) for line in csvreader]

def read_test_labels(tests_file):
    def handle_line(line):
        ans, _ = line.split()
        ans = ans.lower()
        if ans == "true": return True
        if ans == "false": return False
        return None

    with open(tests_file) as test_in:
        return list(handle_line(line) for line in test_in)
