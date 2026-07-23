import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. DADOS DE EXEMPLO (fictícios)
# ------------------------------------------------------------

products = pd.DataFrame({
    'name': [
        'Coca-Cola 350ml', 'Água Mineral 500ml', 'Batata Frita Lays',
        'Chocolate Kit Kat', 'Guaraná Antarctica', 'Amendoim Torrado',
        'Sumo de Laranja Natural', 'Bolacha Recheada', 'Pipoca Doce',
        'Chá Gelado Limão'
    ],
    'category': [
        'Bebida', 'Bebida', 'Snack', 'Doce', 'Bebida',
        'Snack', 'Bebida', 'Doce', 'Snack', 'Bebida'
    ],
    'tags': [
        'refrigerante gelado doce carbonatado',
        'agua natural saudavel sem acucar',
        'salgado crocante batata frita',
        'doce chocolate sobremesa wafer',
        'refrigerante gelado guarana carbonatado',
        'salgado proteina snack saudavel',
        'sumo natural fruta vitamina saudavel',
        'doce bolacha chocolate sobremesa',
        'salgado doce snack milho',
        'bebida gelado cha natural limao'
    ]
})

print("=" * 60)
print("CATÁLOGO DE PRODUTOS")
print("=" * 60)
print(products[['name', 'category']].to_string(index=False))

# ------------------------------------------------------------
# 2. VETORIZAÇÃO (TF-IDF)
# ------------------------------------------------------------

vectorizer = TfidfVectorizer()
combined_text = products['category'] + " " + products['tags']
tfidf_matrix = vectorizer.fit_transform(combined_text)

# ------------------------------------------------------------
# 3. MATRIZ DE SIMILARIDADE
# ------------------------------------------------------------

similarity_matrix = cosine_similarity(tfidf_matrix)

# ------------------------------------------------------------
# 4. FUNÇÃO DE RECOMENDAÇÃO (content-based)
# ------------------------------------------------------------

def recommend_by_product(product_name, top_n=3):
    """Recomenda produtos parecidos com o produto indicado (pelo nome)."""
    idx = products[products['name'] == product_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n + 1]  # ignora o próprio produto

    results = []
    for i, score in scores:
        results.append({
            'name': products.iloc[i]['name'],
            'category': products.iloc[i]['category'],
            'similarity_score': round(score, 3)
        })
    return results


def recommend_by_history(product_names, top_n=3):
    """Recomenda produtos com base em vários produtos que o
    utilizador já comprou (simula um histórico de compras)."""
    indices = products[products['name'].isin(product_names)].index
    average_scores = similarity_matrix[indices].mean(axis=0)

    scores = list(enumerate(average_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # remove produtos já comprados
    scores = [s for s in scores if products.iloc[s[0]]['name'] not in product_names]
    scores = scores[:top_n]

    results = []
    for i, score in scores:
        results.append({
            'name': products.iloc[i]['name'],
            'category': products.iloc[i]['category'],
            'similarity_score': round(score, 3)
        })
    return results


# ------------------------------------------------------------
# 5. TESTES / RESULTADOS DA DEMO
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TESTE 1 — Recomendação a partir de UM produto")
print("=" * 60)
test_product = "Coca-Cola 350ml"
print(f"Utilizador viu/comprou: '{test_product}'")
print("Recomendamos:")
for r in recommend_by_product(test_product):
    print(f"  → {r['name']} ({r['category']}) | similaridade: {r['similarity_score']}")

print("\n" + "=" * 60)
print("TESTE 2 — Recomendação a partir de HISTÓRICO de compras")
print("=" * 60)
test_history = ["Água Mineral 500ml", "Sumo de Laranja Natural"]
print(f"Histórico do utilizador: {test_history}")
print("Recomendamos:")
for r in recommend_by_history(test_history):
    print(f"  → {r['name']} ({r['category']}) | similaridade: {r['similarity_score']}")
