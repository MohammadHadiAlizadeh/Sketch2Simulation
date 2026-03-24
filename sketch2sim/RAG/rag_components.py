import re
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_pure_component_docs_from_excel(excel_path: str | Path) -> list[Document]:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)

    required_cols = {"component_name", "full_name", "formula"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"components_list.xlsx is missing required columns: {missing}")

    docs: list[Document] = []

    for _, row in df.iterrows():
        name = str(row["component_name"]).strip()
        full_name = str(row["full_name"]).strip()
        formula = str(row["formula"]).strip()

        text = (
            "Type: pure_component\n"
            f"HYSYS_name: {name}\n"
            f"Full_name: {full_name}\n"
            f"Formula: {formula}\n"
            f"Aliases: {name}, {full_name}, {formula}\n"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "kind": "pure_component",
                    "component_name": name,
                    "full_name": full_name,
                    "formula": formula,
                },
            )
        )

    return docs


def load_mixture_recipe_docs(txt_path: str | Path) -> list[Document]:
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Mixture recipe file not found: {txt_path}")

    raw_text = txt_path.read_text(encoding="utf-8")

    blocks = [
        block.strip()
        for block in raw_text.split("========================================")
        if block.strip()
    ]

    docs: list[Document] = []
    for block in blocks:
        mixture_name = None
        for line in block.splitlines():
            line = line.strip()
            if line.lower().startswith("mixture_name:"):
                mixture_name = line.split(":", 1)[1].strip()
                break

        docs.append(
            Document(
                page_content=block,
                metadata={
                    "kind": "pseudo_recipe",
                    "mixture_name": mixture_name or "UNKNOWN_MIXTURE",
                },
            )
        )

    return docs


class ExactThenVectorRetriever:
    """
    Hybrid retriever:
    1. Exact lookup over key fields
    2. Fall back to vector retrieval if no exact hit
    """

    def __init__(self, vector_retriever, docs: list[Document]):
        self.vector_retriever = vector_retriever
        self._exact_index: dict[str, list[Document]] = {}
        self._build_exact_index(docs)

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _field(text: str, field: str) -> str:
        match = re.search(rf"(?im)^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", text or "")
        return match.group(1).strip() if match else ""

    def _aliases(self, text: str) -> list[str]:
        aliases = self._field(text, "Aliases")
        if not aliases:
            return []
        return [alias.strip() for alias in aliases.split(",") if alias.strip()]

    def _add_key(self, key: str, doc: Document) -> None:
        normalized_key = self._norm(key)
        if not normalized_key:
            return
        self._exact_index.setdefault(normalized_key, []).append(doc)

    def _build_exact_index(self, docs: list[Document]) -> None:
        for doc in docs:
            text = doc.page_content or ""
            self._add_key(self._field(text, "HYSYS_name"), doc)
            self._add_key(self._field(text, "Full_name"), doc)
            self._add_key(self._field(text, "Formula"), doc)
            self._add_key(self._field(text, "Mixture_name"), doc)
            for alias in self._aliases(text):
                self._add_key(alias, doc)

    @staticmethod
    def _dedupe_keep_order(docs: list[Document]) -> list[Document]:
        seen = set()
        unique_docs: list[Document] = []

        for doc in docs:
            key = (doc.page_content or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)

        return unique_docs

    def invoke(self, query: str, **kwargs) -> list[Document]:
        normalized_query = self._norm(query)

        exact_matches = (
            self._exact_index.get(normalized_query, []) if normalized_query else []
        )
        if exact_matches:
            vector_matches = self.vector_retriever.invoke(query, **kwargs) or []
            return self._dedupe_keep_order(exact_matches + vector_matches)

        vector_matches = self.vector_retriever.invoke(query, **kwargs) or []
        return self._dedupe_keep_order(vector_matches)


def build_component_retriever(
    excel_path: str | Path = "RAG/components_list.xlsx",
    mixture_path: str | Path = "RAG/mixture_recipes.txt",
    persist_dir: str | None = "RAG/chroma_components",
):
    excel_path = Path(excel_path)
    mixture_path = Path(mixture_path)

    pure_docs = load_pure_component_docs_from_excel(excel_path)
    mixture_docs = load_mixture_recipe_docs(mixture_path)
    all_docs = pure_docs + mixture_docs

    if not all_docs:
        raise ValueError("No documents loaded for RAG index.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if persist_dir is not None:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            all_docs,
            embedding=embeddings,
            persist_directory=str(persist_path),
        )
    else:
        vector_store = Chroma.from_documents(all_docs, embedding=embeddings)

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return ExactThenVectorRetriever(vector_retriever, all_docs)
