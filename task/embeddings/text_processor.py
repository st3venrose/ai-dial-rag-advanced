from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)
    def process_text_file(
        self,
        file_path: str,
        chunk_size: int,
        overlap: int,
        dimensions: int,
        truncate_table: bool = False,
    ) -> None:
        """
        Load text from file, chunk it, generate embeddings and store them in DB.
        """
        document_name = file_path.split("/")[-1].split("\\")[-1]

        if truncate_table:
            self._truncate_table()

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content, chunk_size, overlap)
        if not chunks:
            return

        embeddings_map = self.embeddings_client.get_embeddings(chunks, dimensions=dimensions)

        for idx, chunk in enumerate(chunks):
            embedding = embeddings_map.get(idx)
            if embedding is None:
                continue
            self._save_chunk(document_name=document_name, text=chunk, embedding=embedding)

    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`
    def search(
        self,
        mode: SearchMode,
        query: str,
        top_k: int,
        min_score: float,
        dimensions: int,
    ) -> list[str]:
        """
        Generate embedding for the query and search similar chunks in DB.
        """
        if not query:
            return []

        embedding_map = self.embeddings_client.get_embeddings([query], dimensions=dimensions)
        if 0 not in embedding_map:
            return []

        embedding = embedding_map[0]
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        if mode == SearchMode.EUCLIDIAN_DISTANCE:
            distance_operator = "<->"
        else:
            distance_operator = "<=>"

        select_query = f"""
            SELECT text,
                   embedding {distance_operator} %s::vector AS distance
            FROM vectors
            WHERE embedding {distance_operator} %s::vector <= %s
            ORDER BY distance
            LIMIT %s
        """

        chunks: list[str] = []
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (embedding_str, embedding_str, min_score, top_k))
                rows = cursor.fetchall()
                for row in rows:
                    chunks.append(row["text"])

        return chunks

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self) -> None:
        """Remove all existing vectors from the table."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors")

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]) -> None:
        """Persist a single chunk and its embedding in the database."""
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        query = """
            INSERT INTO vectors (document_name, text, embedding)
            VALUES (%s, %s, %s::vector)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (document_name, text, embedding_str))
