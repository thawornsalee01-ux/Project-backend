from src.RAG.embed_query import QueryEmbedder
from src.RAG.chroma_query import ChromaRetriever
from src.RAG.rag_llm import RAGLLM


if __name__ == "__main__":
    question = "การลดค่าปรับอยู่ขั้นใดในกฏหมาย?"

    embedder = QueryEmbedder()
    retriever = ChromaRetriever()
    llm = RAGLLM()

    query_embedding = embedder.embed(question)

    results = retriever.search(
        query_embedding=query_embedding,
        top_k=5,
        document_key="กฏหมาย",
    )

    contexts = results["documents"][0]

    answer = llm.answer(question, contexts)
    print("\n🤖 ANSWER:\n", answer)
