import random
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn import preprocessing
from transformers import BertTokenizer
from collections import Counter


class DataManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.max_sentence_len = 0
        self.str_questions = list()
        self.str_labels = list()
        self.numeral_labels = list()
        self.numeral_data = list()

    def read_csv_data(self, file_path):
        df = pd.read_csv(file_path)

        # Separate 'dual' class
        self.dual_data = df[df["category"] == "dual"]
        non_dual_data = df[df["category"] != "dual"]

        self.str_questions = non_dual_data["word"].tolist()
        self.str_labels = non_dual_data["category"].tolist()
        self.dual_questions = self.dual_data["word"].tolist()

        # Process max sentence length
        for question in self.str_questions + self.dual_questions:
            if self.max_sentence_len < len(str(question)):
                self.max_sentence_len = len(str(question))

        # Encode labels (excluding 'dual')
        le = preprocessing.LabelEncoder()
        le.fit(self.str_labels)
        self.numeral_labels = np.array(le.transform(self.str_labels))
        self.str_classes = le.classes_
        self.num_classes = len(self.str_classes)

        if self.verbose:
            print("\nSample words and corresponding categories (non-dual)... \n")
            print(self.str_questions[:5])
            print(self.str_labels[:5])
            print("\nSample dual words...\n")
            print(self.dual_questions[:5])

    def manipulate_data(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab = self.tokenizer.get_vocab()
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        num_seqs = []
        for text in self.str_questions:
            text_seqs = self.tokenizer.tokenize(str(text))
            token_ids = self.tokenizer.convert_tokens_to_ids(text_seqs)
            seq_tensor = torch.LongTensor(token_ids)
            num_seqs.append(seq_tensor)

        if num_seqs:
            self.numeral_data = pad_sequence(num_seqs, batch_first=True)
            self.num_sentences, self.max_seq_len = self.numeral_data.shape

    def tokenize_data(self, text):
        text_seqs = self.tokenizer.tokenize(str(text))
        token_ids = self.tokenizer.convert_tokens_to_ids(text_seqs)
        seq_tensor = torch.LongTensor(token_ids)
        return seq_tensor

    def train_valid_test_split(self, train_ratio=0.8, test_ratio=0.1):
        # Split non-dual data
        train_size = int(len(self.str_questions) * train_ratio)
        test_size = int(len(self.str_questions) * test_ratio)
        valid_size = len(self.str_questions) - (train_size + test_size)

        indices = list(range(len(self.str_questions)))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        test_indices = indices[train_size : train_size + test_size]
        valid_indices = indices[train_size + test_size :]

        # Create datasets
        train_data = [self.numeral_data[i] for i in train_indices]
        train_labels = [self.numeral_labels[i] for i in train_indices]
        test_data = [self.numeral_data[i] for i in test_indices]
        test_labels = [self.numeral_labels[i] for i in test_indices]
        valid_data = [self.numeral_data[i] for i in valid_indices]
        valid_labels = [self.numeral_labels[i] for i in valid_indices]

        class_counts = Counter(train_labels)
        class_weights = {i: 1.0 / class_counts[i] for i in class_counts}
        weights = [class_weights[i] for i in train_labels]
        sampler = WeightedRandomSampler(weights, len(weights))

        # Create DataLoaders
        self.train_loader = DataLoader(
            TensorDataset(torch.stack(train_data), torch.tensor(train_labels)),
            batch_size=64,
            sampler=sampler,
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.stack(test_data), torch.tensor(test_labels)),
            batch_size=64,
            shuffle=False,
        )
        self.valid_loader = DataLoader(
            TensorDataset(torch.stack(valid_data), torch.tensor(valid_labels)),
            batch_size=64,
            shuffle=False,
        )

        # Create separate loader for dual data
        dual_data = [self.numeral_data[i] for i in range(len(self.dual_questions))]
        self.dual_loader = DataLoader(
            TensorDataset(torch.stack(dual_data)), batch_size=64, shuffle=False
        )
