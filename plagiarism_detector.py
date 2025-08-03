# plagiarism_detector.py

import os
import re
import pickle
import numpy as np
import faiss
import torch.multiprocessing as mp

# Prevent semaphore leaks on macOS
mp.set_sharing_strategy('file_system')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from text_processor import extract_text_from_file, clean_up_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PlagiarismDetector:
    """Combined semantic (FAISS) + TF-IDF fallback plagiarism detector."""

    def __init__(
        self,
        ref_folder="reference_texts",
        index_path="plag_index.faiss",
        mapping_path="mapping.pkl",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        # Set up TF-IDF fallback
        self.text_analyzer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1,3))
        self.reference_texts = [
            "Academic integrity is important in education. Students should do their own work.",
            "The scientific method involves observation, hypothesis, and experimentation.",
            "Climate change is caused by human activities and greenhouse gases.",
            "Technology has changed how we communicate and learn."
        ]

        # Set up sentence-transformers + FAISS
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.ref_folder = ref_folder

        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, "rb") as f:
                self.mapping = pickle.load(f)
        else:
            self.index, self.mapping = self.build_index(self.ref_folder)
            faiss.write_index(self.index, self.index_path)
            with open(self.mapping_path, "wb") as f:
                pickle.dump(self.mapping, f)

        print(f"Semantic index contains {len(self.mapping)} passages.")

    def build_index(self, folder):
        texts, mapping = [], []
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                try:
                    raw = extract_text_from_file(path)
                except Exception:
                    continue
                clean = clean_up_text(raw)
                words = clean.split()
                stride, window = 150, 200
                for i in range(0, max(len(words)-stride,1), stride):
                    chunk = words[i:i+window]
                    if len(chunk) < 50:
                        continue
                    passage = " ".join(chunk)
                    texts.append(passage)
                    mapping.append((fname, passage))

        if not texts:
            # Empty index
            dim = self.model.get_sentence_embedding_dimension()
            return faiss.IndexFlatIP(dim), []

        embs = self.model.encode(texts, convert_to_numpy=True,
                                 show_progress_bar=True, batch_size=32,
                                 device="cpu", normalize_embeddings=True, num_workers=0)
        faiss.normalize_L2(embs)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        return index, mapping

    def _tfidf_similarity(self, submitted):
        clean1 = self._clean_text(submitted)
        best = 0.0
        for ref in self.reference_texts:
            clean2 = self._clean_text(ref)
            vecs = self.text_analyzer.fit_transform([clean1, clean2])
            sim = cosine_similarity(vecs)[0,1] * 100
            best = max(best, sim)
        return round(best,1)

    def _clean_text(self, text):
        t = text.lower()
        t = re.sub(r'\([^)]*\d{4}[^)]*\)', '', t)
        t = re.sub(r'\s+', ' ', t)
        return t.strip()

    def check_for_plagiarism(self, submitted_text, threshold=0.6, top_k=5):
        # First attempt semantic FAISS matching
        clean = clean_up_text(submitted_text)
        words = clean.split()
        if len(self.mapping)==0 or len(words)<50:
            # Fallback to TF-IDF
            score = self._tfidf_similarity(submitted_text)
            confidence = 'high' if score>80 else 'medium' if score>50 else 'low'
            return {
                "overall_score": score,
                "confidence": confidence,
                "suspicious_parts": [],
                "message": "Used TF-IDF fallback (no semantic refs)"
            }

        # Semantic FAISS path
        stride, window = 150, 200
        chunks = []
        for i in range(0, max(len(words)-stride,1), stride):
            chunk = words[i:i+window]
            if len(chunk)>=50:
                chunks.append(" ".join(chunk))

        if not chunks:
            return {"overall_score":0.0,"confidence":"low","suspicious_parts":[],"message":"Too short"}

        embs = self.model.encode(chunks, convert_to_numpy=True,
                                 show_progress_bar=False, batch_size=32,
                                 device="cpu", normalize_embeddings=True, num_workers=0)
        faiss.normalize_L2(embs)

        matches, hit_chunks = [], set()
        for idx, emb in enumerate(embs):
            D,I = self.index.search(emb.reshape(1,-1), top_k)
            print(f"Chunk {idx} sims: {D[0].tolist()}")
            for score,i_ref in zip(D[0],I[0]):
                if score>=threshold:
                    fname, ref_passage = self.mapping[i_ref]
                    matches.append({
                        "submitted_chunk":chunks[idx],
                        "reference_file":fname,
                        "reference_chunk":ref_passage,
                        "similarity":round(float(score),3)
                    })
                    hit_chunks.add(idx)

        overall = 100.0 * len(hit_chunks)/len(chunks)
        confidence = 'high' if overall>80 else 'medium' if overall>50 else 'low'
        return {
            "overall_score":round(overall,1),
            "confidence":confidence,
            "suspicious_parts":matches,
            "message":f"Semantic match against {len(self.mapping)} passages"
        }
