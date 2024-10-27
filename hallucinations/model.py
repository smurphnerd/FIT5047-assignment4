import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import gensim.downloader as api


@dataclass
class Config:
    vocab_size: int
    n_features: int
    n_hidden: int
    word2idx: dict

    # We optimize n_instances models in a single training loop
    # to let us sweep over sparsity or importance curves
    # efficiently.

    # We could potentially use torch.vmap instead.
    # n_instances: int


class Model(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.embed_model = "glove-wiki-gigaword-100"
        self.word2vect = api.load(self.embed_model)
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")
        self.embed_path = "embeddings/E.npy"
        self.embed_size = int(self.embed_model.split("-")[-1])
        self.embed_matrix = np.zeros(shape=[config.vocab_size, self.embed_size])
        self.word2idx = config.word2idx
        self.build_embedding_matrix()

        self.embedding = nn.Embedding.from_pretrained(self.embed_matrix, freeze=True)
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_size,
            out_channels=config.n_features,
            kernel_size=3,
            padding=1,
        )

        self.W = nn.Parameter(torch.empty((config.n_features, config.n_hidden)))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(torch.zeros((config.n_features)))

        # if feature_probability is None:
        #   feature_probability = torch.ones(())
        # self.feature_probability = feature_probability.to(device)
        # if importance is None:
        #   importance = torch.ones(())
        # self.importance = importance.to(device)

    def build_embedding_matrix(self):
        # Insert your code here. Your code should allow for saving the embedding matrix in ``self.embed_path'' (as numpy array) for future retrieval.
        if os.path.exists(self.embed_path):
            self.embed_matrix = np.load(self.embed_path)
        else:
            # Create the embedding matrix with word2vect model
            for word, idx in self.word2idx.items():
                try:
                    self.embed_matrix[idx] = self.word2vect.get_vector(word)
                except KeyError:
                    self.embed_matrix[idx] = np.zeros([self.embed_size])

            # Save the new embedding matrix
            np.save(self.embed_path, self.embed_matrix)

        self.embed_matrix = torch.tensor(self.embed_matrix, dtype=torch.float32)

    def forward(self, x, return_hidden=False):
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = F.relu(x)

        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        hidden = x @ self.W
        out = hidden @ self.W.T + self.b_final
        out = F.relu(out)
        if return_hidden:
            return out, hidden
        return out
