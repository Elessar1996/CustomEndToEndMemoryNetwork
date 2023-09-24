import nltk
import re
import os
import numpy as np





class BagOfTheWordsAggregate:

    def __init__(self, data_directory='data/data/en/', task_id=[15],
                 max_story_length=50, max_query_length=50):

        """
        A class that takes the data of bAbI dataset, based on the task_id,
        parse its data, and embed it using Bag Of the Words method.
        :param data_directory: address in which data located.
        :param task_id: id of the task.

        In order to get the one-hot encoded data, use get_data method:

        obj = BagofWordsbAbI(data_directory='some_directory', task_id=some_id)
        one_hot_encoded_train, one_hot_encoded_test = obj.get_data()

        each output is a tuple consisting of 3 elements: stories, queries, answers

        """
        self.max_story_length = max_story_length
        self.max_query_length = max_query_length
        self.data_directory = data_directory
        self.task_id = task_id
        self.train_data_filename, self.test_data_filename = self.get_file_name()
        self.parsed_train_data = self.parse_stories(self.train_data_filename)
        self.parsed_test_data = self.parse_stories(self.test_data_filename)
        self.word_freq_dict = self.create_vocabulary()
        self.vocabulary_size = len(self.word_freq_dict)
        self.vocabulary = sorted(self.word_freq_dict, key=self.word_freq_dict.get, reverse=True)

        self.word2idx = {w: i + 2 for i, w in enumerate(self.vocabulary)}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<OOV>'] = 1
        train_indiced_padded_stories, train_indiced_padded_queries, train_indiced_answers = self.convert_inputs_to_indices(
            dataset=self.parsed_train_data)
        test_indiced_padded_stories, test_indiced_padded_queries, test_indiced_padded_answers = self.convert_inputs_to_indices(
            dataset=self.parsed_test_data)
        self.train_indiced_padded_data = (
            np.array(train_indiced_padded_stories), np.array(train_indiced_padded_queries),
            np.array(train_indiced_answers))
        self.test_indiced_padded_data = (np.array(test_indiced_padded_stories), np.array(test_indiced_padded_queries),
                                         np.array(test_indiced_padded_answers))

        self.train_one_hot_encoded_data = (
            self.one_hot_array_of_arrays(self.train_indiced_padded_data[0]),
            self.one_hot_array_of_arrays(self.train_indiced_padded_data[1]),
            self.one_hot_array(self.train_indiced_padded_data[2])
        )

        self.test_one_hot_encoded_data = (
            self.one_hot_array_of_arrays(self.test_indiced_padded_data[0]),
            self.one_hot_array_of_arrays(self.test_indiced_padded_data[1]),
            self.one_hot_array(self.test_indiced_padded_data[2])
        )

        self.train_idx_data = (
            self.train_indiced_padded_data[0],
            self.train_indiced_padded_data[1],
            self.train_indiced_padded_data[2]
        )

        self.test_idx_data = (
            self.test_indiced_padded_data[0],
            self.test_indiced_padded_data[1],
            self.test_indiced_padded_data[2]
        )

    def concat_items_of_datasets(self, dataset):

        output = []

        stories, queries, answers = dataset


        # reversed_stories = self.reverse_stories(stories)

        assert len(stories) == len(queries) and len(queries) == len(
            answers), 'stories, queries and answers must have same length'

        for story, query, answer in zip(stories, queries, answers):
            output.append((story, query, answer))

        return output

    # def reverse_sentence(self, sentence):
    #
    #     return np.flip(sentence)

    def reverse_story(self, story):

        return np.flip(story, axis=0)

    def reverse_stories(self, stories):

        output = []

        for story in stories:
            output.append(self.reverse_story(story))

        return np.array(output)

    def get_data(self):

        train_data = self.concat_items_of_datasets(self.train_one_hot_encoded_data)

        test_data = self.concat_items_of_datasets(self.test_one_hot_encoded_data)

        return train_data, test_data

    def get_idx_data(self):
      train_data = self.concat_items_of_datasets(self.train_idx_data)
      test_data = self.concat_items_of_datasets(self.test_idx_data)

      return train_data, test_data

    def pad_sentences(self, story_indices, max_length):

      output = []

      for sentence in story_indices:
        s = sentence.copy()
        if len(sentence) > max_length:

          output.append(np.array(s[:max_length]))

        else:

          output.append(np.array(s + (max_length - len(s))*[0]))

      return np.array(output)

    def convert_inputs_to_indices(self, dataset):

        stories = []
        queries = []
        answers = []

        for story, question, answer in dataset:

            story_indices = [self.sentence_to_indices(sentence) for sentence in story]
            story_indices_padded = self.pad_sentences(story_indices, self.max_story_length)
            story_indices_flattened = sum(story_indices, [])
            story_indices_flattened_padded = self.pad_indices(story_indices_flattened, self.max_story_length)

            query_indices = self.sentence_to_indices(question)
            query_indices_padded = self.pad_indices(query_indices, self.max_query_length)

            answer_index = self.word_to_index(answer[0])

            stories.append(story_indices_flattened_padded)
            queries.append(query_indices_padded)
            answers.append(answer_index)

        return stories, queries, answers

    def word_to_index(self, word):
        try:
            return self.word2idx[word]
        except KeyError:
            print(f'word {word} raised a key error. adding data to the dictionary')
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.vocabulary_size = len(self.word2idx)
            print(f'now vocab size: {self.vocabulary_size}')
            return idx

    def sentence_to_indices(self, sentence):

        return [self.word_to_index(word) for word in sentence]

    def pad_indices(self, indices, length):

        if len(indices) >= length:

            return indices[:length]
        else:

            return indices + [0] * (length - len(indices))

    def create_vocabulary(self):

        word_freq = dict()

        total = self.parsed_train_data + self.parsed_test_data


        for idx, (story, query, _) in enumerate(total):
            for words in story:
                for word in words:

                    if word not in list(word_freq.keys()):
                        word_freq[word] = 1
                    else:
                        word_freq[word] += 1
            for word in query:


                if word not in list(word_freq.keys()):
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        return word_freq

    def parse_stories(self, filenames):

        stories = []
        story = []

        for fn in filenames:
          with open(fn, 'r') as f:

              lines = f.readlines()

          for idx, line in enumerate(lines):

              nid, line = line.split(" ", 1)
              nid = int(nid)

              if nid == 1:
                  story = []

              if '\t' in line:

                  query, answer, _ = line.split('\t')

                  query = nltk.tokenize.word_tokenize(query.lower())
                  answer = [answer.lower()]

                  stories.append((story.copy(), query, answer))

              else:
                  sentence = nltk.tokenize.word_tokenize(line.lower())

                  story.append(sentence)

        return stories

    def get_file_name(self):

        files = os.listdir(self.data_directory)

        output = []

        train = []

        test = []

        for file in files:

            numeric_part = re.findall(r'\d+', file)

            numeric_part = int(numeric_part[0])

            if numeric_part in self.task_id:
                output.append(file)
        print(f'output: {output}')
        for f in output:

            if 'train' in f:

                train.append(f)
            else:
                test.append(f)

        train_directories = [os.path.join(self.data_directory, t) for t in train]
        test_directories = [os.path.join(self.data_directory, t) for t in test]

        print(f'train directories: {train_directories}')
        print(f'test directories: {test_directories}')
        return train_directories, test_directories

    def one_hot_array(self, array):

        output = []
        for item in array:
            zeros = np.zeros(shape=(self.vocabulary_size + 2,))
            zeros = zeros.tolist()
            zeros[item] = 1
            zeros = np.array(zeros)
            output.append(zeros)

        return np.array(output)

    def one_hot_array_of_arrays(self, array_of_arrays):

        output = []

        for array in array_of_arrays:
            output.append(self.one_hot_array(array))

        return np.array(output)

    def one_hot_array_of_array_of_arrays(self, array_of_array_of_arrays):

        output = []

        for array in array_of_array_of_arrays:
            output.append(self.one_hot_array_of_arrays(array))

        return np.array(output)

