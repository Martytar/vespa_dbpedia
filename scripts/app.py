import json

from tqdm import tqdm
from vespa.application import Vespa
from vespa.deployment import VespaDocker
from vespa.package import ApplicationPackage, Document, Field, FieldSet, HNSW, Schema, RankProfile
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time


class ColBERTEncoder:
    def __init__(self):
        print("Loading ColBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("vespa-engine/colbert-medium")
        self.model = AutoModel.from_pretrained("vespa-engine/colbert-medium")
        print("ColBERT model loaded successfully")

    def encode(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # ColBERT-medium имеет размерность 512, используем полную размерность
        print(f"Embedding length: {len(embeddings)}")
        return embeddings.tolist()


class VespaNewsApp:
    def __init__(self):
        self.vespa_docker = None
        self.app = None
        self.colbert_encoder = ColBERTEncoder()

    def create_application_package(self):
        """Создает ApplicationPackage программно"""

        # Обновляем размерность тензора на 512 (размерность ColBERT-medium)
        document = Document(
            fields=[
                Field(name="news_id", type="string", indexing=["summary", "attribute"]),
                Field(name="title", type="string", indexing=["index", "summary"], index="enable-bm25"),
                Field(name="description", type="string", indexing=["index", "summary"], index="enable-bm25"),
                Field(name="title_embeddings", type="tensor<float>(d0[512])",
                      indexing=["attribute", "summary", "index"],
                      ann=HNSW(distance_metric="innerproduct")),
                Field(name="description_embeddings", type="tensor<float>(d0[512])",
                      indexing=["attribute", "summary", "index"],
                      ann=HNSW(distance_metric="innerproduct"))
            ]
        )

        # Создаем схему с правильными inputs
        news_schema = Schema(
            name="news",
            document=document,
            fieldsets=[FieldSet(name="default", fields=["title", "description"])],
            rank_profiles=[
                RankProfile(
                    name="bm25",
                    inherits="default",
                    first_phase="bm25(title) + bm25(description)"
                ),
                RankProfile(
                    name="colbert",
                    inherits="default",
                    inputs=[("query(query_embedding)", "tensor<float>(d0[512])")],
                    first_phase="closeness(title_embeddings)"
                ),
                RankProfile(
                    name="hybrid",
                    inherits="default",
                    inputs=[("query(query_embedding)", "tensor<float>(d0[512])")],
                    first_phase="(bm25(title) + bm25(description)) * 0.6 + closeness(title_embeddings) * 0.4"
                )
            ]
        )

        # Создаем пакет приложения
        app_package = ApplicationPackage(
            name="news",
            schema=[news_schema]
        )

        return app_package

    def deploy(self):
        """Развертывание приложения в Docker"""
        print("Creating Vespa application package...")
        app_package = self.create_application_package()

        print("Starting Vespa Docker container...")
        self.vespa_docker = VespaDocker(
            port=8080,
            container_memory="4G"
        )

        # Загружаем приложение
        self.app = self.vespa_docker.deploy(
            application_package=app_package
        )

        print("Vespa application deployed successfully!")
        return self.app

    def wait_for_application_ready(self, timeout: int = 120):
        """Ожидает пока приложение будет готово"""
        print("Waiting for application to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.app.get_application_status()
                if response.status_code == 200:
                    print("Application is ready!")
                    return True
            except Exception as e:
                print(f"Waiting... {e}")
            time.sleep(5)
        raise Exception("Application failed to start within timeout")

    def index_document(self, news_id: str, title: str, description: str):
        """Индексация документа с ColBERT эмбеддингами"""
        print(f"Indexing document {news_id}: {title}")

        # Генерация эмбеддингов
        title_embeddings = self.colbert_encoder.encode(title)
        description_embeddings = self.colbert_encoder.encode(description)

        # Проверим длину эмбеддингов
        print(f"Title embeddings length: {len(title_embeddings)}")
        print(f"Description embeddings length: {len(description_embeddings)}")

        # Используем упрощенный формат с values
        document = {
            "fields": {
                "news_id": news_id,
                "title": title,
                "description": description,
                "title_embeddings": {
                    "values": title_embeddings
                },
                "description_embeddings": {
                    "values": description_embeddings
                }
            }
        }

        response = self.app.feed_data_point(
            schema="news",
            data_id=news_id,
            fields=document["fields"]
        )

        print(f"Document {news_id} indexed with status: {response.status_code}")
        return response

    def search_bm25(self, query: str, hits: int = 10):
        """Поиск с использованием BM25"""
        print(f"BM25 search: '{query}'")
        return self.app.query(
            yql="select * from news where userQuery()",
            query=query,
            ranking="bm25",
            hits=hits
        )

    def search_colbert(self, query: str, hits: int = 10):
        """Поиск с использованием ColBERT"""
        print(f"ColBERT search: '{query}'")
        query_embeddings = self.colbert_encoder.encode(query)

        # Правильный формат для query тензора
        return self.app.query(
            yql="select * from news where ({targetHits:100}nearestNeighbor(title_embeddings, query_embedding))",
            ranking="colbert",
            body={
                "input": {
                    "query(query_embedding)": query_embeddings
                }
            },
            hits=hits
        )

    def search_hybrid(self, query: str, hits: int = 10):
        """Гибридный поиск BM25 + ColBERT"""
        print(f"Hybrid search: '{query}'")
        query_embeddings = self.colbert_encoder.encode(query)

        return self.app.query(
            yql="select * from news where userQuery() or ({targetHits:100}nearestNeighbor(title_embeddings, query_embedding))",
            query=query,
            ranking="hybrid",
            body={
                "input": {
                    "query(query_embedding)": query_embeddings
                }
            },
            hits=hits
        )


def print_results(results, search_type):
    """Печатает результаты поиска"""
    print(f"\n=== {search_type} Search Results ===")
    if hasattr(results, 'hits') and results.hits:
        for i, hit in enumerate(results.hits[:5]):
            print(f"{i + 1}. Score: {hit['relevance']:.4f}")
            print(f"   Title: {hit['fields'].get('title', 'N/A')}")
            print(f"   ID: {hit['fields'].get('news_id', 'N/A')}")
            print()
    else:
        print("No results found")
    print("-" * 50)


if __name__ == "__main__":
    try:
        # Создаем и развертываем приложение
        news_app = VespaNewsApp()
        app = news_app.deploy()

        # Ждем пока приложение запустится
        news_app.wait_for_application_ready(timeout=120)

        # Выполняем поиск
        query = "AI healthcare"

        # BM25 поиск
        bm25_results = news_app.search_bm25(query)
        print_results(bm25_results, "BM25")

        # ColBERT поиск
        colbert_results = news_app.search_colbert(query)
        print_results(colbert_results, "ColBERT")

        # Гибридный поиск
        hybrid_results = news_app.search_hybrid(query)
        print_results(hybrid_results, "Hybrid")
        print("Demo completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()