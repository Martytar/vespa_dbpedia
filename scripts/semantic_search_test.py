import requests
import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class ColBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vespa-engine/colbert-medium")
        self.model = AutoModel.from_pretrained("vespa-engine/colbert-medium")

    def encode(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings.tolist()


class RankProfileTester:
    def __init__(self, vespa_url: str = "http://localhost:8080"):
        self.vespa_url = vespa_url
        self.encoder = ColBERTEncoder()

    def test_bm25_profile(self, query: str, hits: int = 2):
        """Тестирование rank-profile bm25"""
        url = f"{self.vespa_url}/search/"
        params = {
            "yql": "select * from news where userQuery()",
            "query": query,
            "ranking": "bm25",
            "hits": hits
        }
        response = requests.get(url, params=params)
        return response.json()

    def test_colbert_profile(self, query: str, hits: int = 2):
        """Тестирование rank-profile colbert"""
        query_embeddings = self.encoder.encode(query)

        url = f"{self.vespa_url}/search/"
        params = {
            "yql": "select * from news where ({targetHits:100}nearestNeighbor(title_embeddings, query_embedding))",
            "ranking": "colbert",
            "hits": hits
        }

        body = {
            "input": {
                "query(query_embedding)": query_embeddings
            }
        }

        response = requests.post(url, params=params, json=body)
        return response.json()

    def test_hybrid_profile(self, query: str, hits: int = 2):
        """Тестирование rank-profile hybrid"""
        query_embeddings = self.encoder.encode(query)

        url = f"{self.vespa_url}/search/"
        params = {
            "yql": "select * from news where userQuery() or ({targetHits:100}nearestNeighbor(title_embeddings, query_embedding))",
            "query": query,
            "ranking": "hybrid",
            "hits": hits
        }

        body = {
            "input": {
                "query(query_embedding)": query_embeddings
            }
        }

        response = requests.post(url, params=params, json=body)
        return response.json()

    def test_default_profile(self, query: str, hits: int = 2):
        """Тестирование базового rank-profile default"""
        url = f"{self.vespa_url}/search/"
        params = {
            "yql": "select * from news where userQuery()",
            "query": query,
            "ranking": "default",
            "hits": hits
        }
        response = requests.get(url, params=params)
        return response.json()

    def print_profile_results(self, results, profile_name, query):
        """Вывод результатов для конкретного рангового профиля"""
        print(f"\n🎯 {profile_name}")
        print(f"   Запрос: '{query}'")
        print("-" * 50)

        if 'root' in results and 'children' in results['root']:
            hits = results['root']['children']
            total_count = results['root']['fields'].get('totalCount', 0)

            print(f"   Найдено документов: {total_count}")

            for i, hit in enumerate(hits, 1):
                fields = hit['fields']
                print(f"   {i}. Score: {hit['relevance']:.4f}")
                print(f"      Title: {fields.get('title', 'N/A')}")
                if fields.get('description'):
                    print(f"      Desc: {fields.get('description', '')[:80]}...")
                print(f"      ID: {fields.get('news_id', 'N/A')}")
        else:
            print("   No results found")

        print()


def main():
    tester = RankProfileTester()

    # Тестовые запросы для разных сценариев
    test_cases = [
        {
            "name": "Точное текстовое совпадение",
            "queries": ["Wall Street stocks", "Reuters news", "NASA artificial intelligence"]
        },
        {
            "name": "Семантические запросы",
            "queries": ["machine learning healthcare", "data science medicine", "AI technology innovations"]
        },
        {
            "name": "Гибридные запросы",
            "queries": ["artificial intelligence in medical diagnosis", "deep learning for healthcare",
                        "neural networks in finance"]
        }
    ]

    for test_case in test_cases:
        print(f"\n{'=' * 80}")
        print(f"📋 ТЕСТ: {test_case['name']}")
        print(f"{'=' * 80}")

        for query in test_case['queries']:
            print(f"\n🔍 Запрос: '{query}'")
            print("=" * 60)

            # Тестируем все ранговые профили
            print("\n📊 Ранговые профили:")
            print("-" * 40)

            # 1. Default profile
            default_results = tester.test_default_profile(query)
            tester.print_profile_results(default_results, "default", query)

            # 2. BM25 profile
            bm25_results = tester.test_bm25_profile(query)
            tester.print_profile_results(bm25_results, "bm25", query)

            # 3. ColBERT profile
            colbert_results = tester.test_colbert_profile(query)
            tester.print_profile_results(colbert_results, "colbert", query)

            # 4. Hybrid profile
            hybrid_results = tester.test_hybrid_profile(query)
            tester.print_profile_results(hybrid_results, "hybrid", query)

            print("=" * 60)


if __name__ == "__main__":
    main()