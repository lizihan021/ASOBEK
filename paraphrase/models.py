import modelclasses as model
import csv
import re

class InvalidLabelFormat(Exception):
    pass


class TrainDataRow(model.Model):
    topic_id = model.Integer
    s1_id = model.Integer
    s2_id = model.Integer
    sent_1 = model.String
    sent_2 = model.String
    label = model.Integer

    def is_paraphrase(self):
        return self.label

    def is_debatable(self):
        return 0
        #return self.label[0] is 2

class TestDataRow(model.Model):
    topic_id = model.Integer
    sent_1 = model.String
    sent_2 = model.String
    # label = model.Integer

    # def is_paraphrase(self):
    #     return self.label in (4, 5)

    # def is_debatable(self):
    #     return self.label is 3


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
