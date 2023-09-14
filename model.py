import torch
from torch import nn


class EndToEndMemoryNetworkHops(nn.Module):

    def __init__(self, embedding_size, vocabulary_size, memory_size, sentence_size, num_hops=1):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_size = int(embedding_size)
        self.vocabulary_size = int(vocabulary_size) + 2
        self.memory_size = memory_size
        self.sentence_size = sentence_size
        self.num_hops = num_hops
        self.A = nn.Embedding(num_embeddings=sentence_size, embedding_dim=embedding_size)
        self.C = nn.Embedding(num_embeddings=sentence_size, embedding_dim=embedding_size)
        self.W = nn.Linear(in_features=self.embedding_size, out_features=self.vocabulary_size)
        nn.init.normal_(self.W.weight, mean=0, std=0.1)
        nn.init.normal_(self.A.weight, mean=0, std=0.1)
        nn.init.normal_(self.C.weight, mean=0, std=0.1)

        for i in range(num_hops - 1):
            setattr(self, f'C_{i}', nn.Embedding(num_embeddings=sentence_size, embedding_dim=embedding_size))

        for i in range(num_hops - 1):
            nn.init.normal_(getattr(self, f'C_{i}').weight, mean=0, std=0.1)

        self.temporal_encoding_A = nn.Parameter(
            torch.normal(mean=0, std=0.1, size=(self.sentence_size, self.embedding_size)))
        self.temporal_encoding_C = nn.Parameter(
            torch.normal(mean=0, std=0.1, size=(self.sentence_size, self.embedding_size)))

        self.softmax = nn.Softmax(dim=1)

    def get_positional_embedding(self, sentence_size, batch_size):

        positional_embedding = []

        row = []

        for k in range(1, sentence_size + 1):
            for j in range(1, self.embedding_size + 1):
                element = (1 - j / sentence_size) - (k / self.embedding_size) * (
                        1 - 2 * j / sentence_size)
                row.append(element)
            positional_embedding.append(row)
            row = []

        unbatched_positional_embedding = torch.tensor(positional_embedding)

        batched_positional_embedding = unbatched_positional_embedding.expand(batch_size, -1, -1)

        return batched_positional_embedding.to(self.device)

    def forward(self, story, query):

        story = story.to(self.device)
        query = query.to(self.device)

        story = story.long()
        query = query.long()

        batch_size = story.shape[0]
        story_sentence_size = story.shape[1]
        query_sentence_size = query.shape[1]

        story_positional_embedding = self.get_positional_embedding(story_sentence_size, batch_size)
        query_positional_embedding = self.get_positional_embedding(query_sentence_size, batch_size)

        embedded_story = self.A(story)
        embedded_query = self.A(query)
        embedded_story_output = self.C(story)

        embedded_positionaly_encoded_story = torch.mul(embedded_story, story_positional_embedding)
        embedded_positionaly_encoded_query = torch.mul(embedded_query, query_positional_embedding)
        embedded_positionaly_encoded_story_output = torch.mul(embedded_story_output, story_positional_embedding)

        embedded_positionaly_encoded_temporaly_encoded_story = embedded_positionaly_encoded_story + self.temporal_encoding_A.expand(
            batch_size, -1, -1)
        embedded_positionaly_encoded_temporaly_encoded_story_output = embedded_positionaly_encoded_story_output + self.temporal_encoding_C.expand(
            batch_size, -1, -1)
        memory = torch.sum(embedded_positionaly_encoded_temporaly_encoded_story, dim=1)

        c = torch.sum(embedded_positionaly_encoded_temporaly_encoded_story_output, dim=1)

        internal_state_u = torch.sum(embedded_positionaly_encoded_query, dim=1)

        attn = torch.matmul(internal_state_u.unsqueeze(dim=-1), memory.unsqueeze(dim=1))

        attn = nn.Softmax(dim=-1)(attn)
        memory_output = torch.bmm(c.unsqueeze(dim=1), attn)
        memory_output = torch.sum(memory_output, dim=1)

        internal_state_u = internal_state_u.unsqueeze(dim=1) + memory_output.unsqueeze(dim=1)

        for hop in range(self.num_hops - 1):
            if hop == 0:
                story_embedded_in_hops = self.C(story)
            else:
                story_embedded_in_hops = getattr(self, f'C_{hop - 1}')(story)

            embedded_story_output_in_hops = getattr(self, f'C_{hop}')(story)

            embedded_positionaly_encoded_story_in_hops = torch.mul(story_embedded_in_hops, story_positional_embedding)
            embedded_positionaly_encoded_story_output_in_hops = torch.mul(embedded_story_output_in_hops,
                                                                          story_positional_embedding)
            embedded_positionaly_encoded_temporaly_encoded_story_in_hops = embedded_positionaly_encoded_story_in_hops + self.temporal_encoding_A.expand(
                batch_size, -1, -1)
            embedded_positionaly_encoded_temporaly_encoded_story_output_in_hops = embedded_positionaly_encoded_story_output_in_hops + self.temporal_encoding_C.expand(
                batch_size, -1, -1)
            memory_in_hops = torch.sum(embedded_positionaly_encoded_temporaly_encoded_story_in_hops, dim=1)
            c_in_hops = torch.sum(embedded_positionaly_encoded_temporaly_encoded_story_output_in_hops, dim=1)

            attn_in_hops = torch.matmul(memory_in_hops.unsqueeze(dim=-1), internal_state_u)
            attn_in_hops = nn.Softmax(dim=-1)(attn)
            memory_output_in_hops = torch.bmm(c_in_hops.unsqueeze(dim=1), attn_in_hops)
            memory_output_in_hops = torch.sum(memory_output_in_hops, dim=1)
            internal_state_u = internal_state_u + memory_output_in_hops.unsqueeze(dim=1)

        a_hat = self.W(internal_state_u)
        a_hat = nn.Softmax(dim=-1)(a_hat)
        return a_hat
