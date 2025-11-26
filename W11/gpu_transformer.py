# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")

print(f"目前使用的運算裝置: {device}")

# ======================================================
# 1. 建立資料 (raw data)
# ======================================================

raw_data = [
    ("I love AI", "我愛人工智慧"),
    ("Deep learning is fun", "深度學習很有趣"),
    ("Transformer is powerful", "變壓器模型很強大"),
    ("This is a long sentence to test padding mechanism", "這是一個用來測試填充機制的句子"),
    ("GPU makes training faster", "GPU讓訓練變更快"),
    ("Seq2Seq model is cool", "序列到序列模型很酷"),
]

# ======================================================
# 2. Tokenizer  (SimpleTokenizer)
# ======================================================

class SimpleTokenizer:
    def __init__(self, data, lang_idx):
        self.word2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}

        vocab = set()
        for pair in data:
            sentence = pair[lang_idx]
            if lang_idx == 0:       # English → split by space
                words = sentence.split()
            else:                   # Chinese → one char per token
                words = list(sentence)
            vocab.update(words)

        for i, word in enumerate(vocab):
            self.word2idx[word] = i + 4
            self.idx2word[i + 4] = word

    def encode(self, text, lang_type="en"):
        if lang_type == "en":
            words = text.split()
        else:
            words = list(text)
        return [self.word2idx.get(w, 3) for w in words]

    def decode(self, indices):
        return "".join(
            [self.idx2word.get(idx, "") for idx in indices if idx not in [0, 1, 2]]
        )


src_tokenizer = SimpleTokenizer(raw_data, 0)
tgt_tokenizer = SimpleTokenizer(raw_data, 1)

# ======================================================
# 2. Dataset
# ======================================================

class TranslationDataset(Dataset):
    def __init__(self, data, src_tok, tgt_tok):
        self.data = data
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        src_ids = [1] + self.src_tok.encode(src_text, "en") + [2]   # BOS ... EOS
        tgt_ids = [1] + self.tgt_tok.encode(tgt_text, "ch") + [2]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


# ======================================================
# 3. 建立模型 (Seq2SeqTransformer)
# ======================================================

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_mask = (tgt_mask == float('-inf'))


        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.fc_out(outs)


# ======================================================
# 4. 訓練函數
# ======================================================

def train():
    BATCH_SIZE = 2
    EPOCHS = 20
    LR = 0.0001

    dataset = TranslationDataset(raw_data, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_tokenizer.word2idx),
        tgt_vocab_size=len(tgt_tokenizer.word2idx),
        d_model=256,
        nhead=4,
        num_layers=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    print("開始訓練...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Teacher Forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            logits = model(src, tgt_input)

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tgt_output.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")

    return model


# ======================================================
# 5. 推論函數 (translate)
# ======================================================

def translate(model, src_sentence):
    model.eval()

    src_ids = [1] + src_tokenizer.encode(src_sentence, 'en') + [2]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    print(f"\n輸入句子: {src_sentence}")
    print("翻譯結果:", end=" ")

    tgt_ids = [1]   # <BOS>

    for _ in range(20):
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)

        next_token_id = logits[0, -1].argmax().item()
        tgt_ids.append(next_token_id)

        if next_token_id == 2:
            break

    result = tgt_tokenizer.decode(tgt_ids)
    print(result)


# ======================================================
# 主程式
# ======================================================

if __name__ == "__main__":
    trained_model = train()
    translate(trained_model, "I love AI")
    translate(trained_model, "Deep learning is fun")
