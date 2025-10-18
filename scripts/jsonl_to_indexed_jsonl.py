import json
import torch
from transformers import AutoTokenizer, AutoModel


class SimpleColBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vespa-engine/colbert-medium")
        self.model = AutoModel.from_pretrained("vespa-engine/colbert-medium")
        self.model.eval()  # Переводим модель в режим оценки

    def encode(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.tolist()


def simple_process_jsonl(input_file, output_file):
    encoder = SimpleColBERTEncoder()

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                record = json.loads(line.strip())
                fields = record["fields"]

                # Генерируем эмбеддинги
                title_emb = encoder.encode(fields["title"])
                desc_emb = encoder.encode(fields["description"])

                # Добавляем к существующим полям
                fields["title_embeddings"] = {"values": title_emb}
                fields["description_embeddings"] = {"values": desc_emb}

                outfile.write(json.dumps(record) + '\n')
                #print(f"Processed news_id: {fields['news_id']}")

            except Exception as e:
                print(f"Error: {e}")
                # Сохраняем оригинальную запись в случае ошибки
                outfile.write(line)


# Запуск
simple_process_jsonl("news.jsonl", #change to correct paths
                     "news_indexed.jsonl")