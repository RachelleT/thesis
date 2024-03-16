import glob
import random


class Preprocessing:
    def __init__(self):
        self.images_path = "Data/"
        file_list = glob.glob(self.images_path + "*")
        files = sorted(file_list)
        self.data = []
        for class_path in files:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])

    def numbers_generator(self):
        class_total = [0, 104, 135, 424, 793, 799, 880, 1016, 1079, 1581, 1815]
        class_split = [10, 3, 29, 37, 1, 8, 14, 6, 50, 23]
        test_index = []
        for i in range(0, len(class_split)):
            random_list = random.sample(range(class_total[i], class_total[i+1]), class_split[i])
            test_index.append(sorted(random_list))
        return test_index

    def split_data(self):
        train= []
        test = []
        test_indices = self.numbers_generator()
        temp_list = []
        for lst in test_indices:
            for j in lst:
                test.append(self.data[j])
                temp_list.append(j)
        for i in range(0, len(self.data)):
            if i not in temp_list:
                train.append(self.data[i])
        return train, test

        

