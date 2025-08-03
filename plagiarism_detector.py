# plagiarism_detector.py

import os
import re
import pickle
import numpy as np
import faiss
import torch.multiprocessing as mp

# Prevent semaphore leaks on macOS
mp.set_sharing_strategy('file_system')

# Disable Hugging Face tokenizers parallelism to avoid subprocesses
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable OpenMP in FAISS and NumPy/BLAS to prevent threaded resource leaks
faiss.omp_set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
from text_processor import extract_text_from_file, clean_up_text

class PlagiarismDetector:
    """Enhanced plagiarism detector using semantic embeddings + FAISS."""

    def __init__(
        self,
        ref_folder="reference_texts",
        index_path="plag_index.faiss",
        mapping_path="mapping.pkl",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize embedding model, build or load FAISS index."""
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.ref_folder = ref_folder

        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, "rb") as f:
                self.mapping = pickle.load(f)
            print(f"Loaded FAISS index with {len(self.mapping)} passages.")
        else:
            self.index, self.mapping = self.build_index(self.ref_folder)
            faiss.write_index(self.index, self.index_path)
            with open(self.mapping_path, "wb") as f:
                pickle.dump(self.mapping, f)
            print(f"Built FAISS index with {len(self.mapping)} passages.")

    def build_index(self, folder):
        """Extract, chunk, embed, and index all reference documents."""
        texts = []
        mapping = []

        if not os.path.isdir(folder):
            print(f"Reference folder '{folder}' not found; creating empty index.")
            dim = self.model.get_sentence_embedding_dimension()
            return faiss.IndexFlatIP(dim), []

        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            try:
                raw = extract_text_from_file(path)
            except Exception:
                continue
            clean = clean_up_text(raw)
            words = clean.split()

            stride = 150
            window = 200
            for i in range(0, max(len(words) - stride, 1), stride):
                chunk_words = words[i : i + window]
                if len(chunk_words) < 50:
                    continue
                passage = " ".join(chunk_words)
                texts.append(passage)
                mapping.append((fname, passage))

        if not texts:
            print("No passages extracted; index remains empty.")
            dim = self.model.get_sentence_embedding_dimension()
            return faiss.IndexFlatIP(dim), []

        # Single-process embedding
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
            device="cpu",
            normalize_embeddings=True,
            num_workers=0
        )
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return index, mapping

    def check_for_plagiarism(self, submitted_text, threshold=0.6, top_k=5):
        """
        Check a submitted text for similarity against reference corpus.
        Returns an overall score and detailed matches.
        """
        clean = clean_up_text(submitted_text)
        words = clean.split()

        if len(words) < 50 or not hasattr(self, "index"):
            return {
                "overall_score": 0.0,
                "confidence": "low",
                "suspicious_parts": [],
                "message": "No reference corpus available or text too short"
            }

        stride = 150
        window = 200
        chunks = []
        for i in range(0, max(len(words) - stride, 1), stride):
            chunk_words = words[i : i + window]
            if len(chunk_words) >= 50:
                chunks.append(" ".join(chunk_words))

        if not chunks:
            return {
                "overall_score": 0.0,
                "confidence": "low",
                "suspicious_parts": [],
                "message": "Text too short for chunking"
            }

        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
            device="cpu",
            normalize_embeddings=True,
            num_workers=0
        )
        faiss.normalize_L2(embeddings)

        all_matches = []
        chunks_with_hits = set()

        for idx, emb in enumerate(embeddings):
            D, I = self.index.search(emb.reshape(1, -1), top_k)
            # Debug: log top similarity scores
            print(f"Chunk {idx} top similarities: {D[0].tolist()}")
            for score, i_ref in zip(D[0], I[0]):
                if score >= threshold:
                    fname, ref_passage = self.mapping[i_ref]
                    all_matches.append({
                        "submitted_chunk": chunks[idx],
                        "reference_file": fname,
                        "reference_chunk": ref_passage,
                        "similarity": round(float(score), 3)
                    })
                    chunks_with_hits.add(idx)

        overall = 100.0 * (len(chunks_with_hits) / len(chunks))
        confidence = "high" if overall > 80 else "medium" if overall > 50 else "low"

        return {
            "overall_score": round(overall, 1),
            "confidence": confidence,
            "suspicious_parts": all_matches,
            "message": f"Compared against {len(self.mapping)} passages"
        }
