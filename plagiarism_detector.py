# plagiarism_detector.py

import os, re, pickle
import faiss
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
from text_processor import extract_text_from_file, clean_up_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Multiprocessing & OpenMP settings
mp.set_sharing_strategy('file_system')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
faiss.omp_set_num_threads(1)

class PlagiarismDetector:
    """Plagiarism checker with adjustable semantic threshold and TF-IDF fallback."""

    def __init__(
        self,
        ref_folder="reference_texts",
        semantic_threshold=0.4,
        tfidf_refs=None
    ):
        """
        semantic_threshold: cosine similarity cutoff for FAISS matches.
        tfidf_refs: list of fallback reference strings for TF-IDF detection.
        """
        self.threshold = semantic_threshold
        self.tfidf_refs = tfidf_refs or [
            "Academic integrity is important in education. Students should do their own work.",
            "The scientific method involves observation, hypothesis, and experimentation.",
            "Climate change is caused by human activities and greenhouse gases.",
            "Technology has changed how we communicate and learn."
        ]
        self.text_analyzer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1,3))

        # Build or load semantic index
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.ref_folder = ref_folder
        # Use subject-specific index and mapping files
        safe_folder = os.path.basename(os.path.normpath(ref_folder))
        self.index_path = f"plag_index_{safe_folder}.faiss"
        self.mapping_path = f"mapping_{safe_folder}.pkl"

        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, "rb") as f:
                self.mapping = pickle.load(f)
        else:
            self.index, self.mapping = self.build_index(ref_folder)
            faiss.write_index(self.index, self.index_path)
            with open(self.mapping_path, "wb") as f:
                pickle.dump(self.mapping, f)

    def build_index(self, folder):
        texts, mapping = [], []
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                path = os.path.join(folder, fn)
                print(f"[DEBUG] Processing file: {fn}")
                try:
                    raw = extract_text_from_file(path)
                except Exception as e:
                    print(f"[DEBUG] Extraction failed for {fn}: {e}")
                    continue
                clean = clean_up_text(raw)
                words = clean.split()
                print(f"[DEBUG] {fn}: {len(words)} words extracted after cleanup.")
                chunked = 0
                for i in range(0, max(len(words)-150,1), 150):
                    chunk = words[i:i+200]
                    if len(chunk) >= 50:
                        txt = " ".join(chunk)
                        texts.append(txt)
                        mapping.append((fn, txt))
                        chunked += 1
                print(f"[DEBUG] {fn}: {chunked} chunks of >=50 words.")
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return faiss.IndexFlatIP(dim), []
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, num_workers=0)
        faiss.normalize_L2(embs)
        idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
        return idx, mapping

    def _tfidf_similarity(self, text):
        t1 = re.sub(r'\s+', ' ', text.lower()).strip()
        best = 0.0
        for ref in self.tfidf_refs:
            t2 = re.sub(r'\s+', ' ', ref.lower()).strip()
            vecs = self.text_analyzer.fit_transform([t1, t2])
            sim = cosine_similarity(vecs)[0,1] * 100
            best = max(best, sim)
        return round(best,1)

    def check_for_plagiarism(self, submitted_text, top_k=5):
        clean = clean_up_text(submitted_text)
        words = clean.split()
        # If no semantic refs or too short, fallback to TF-IDF
        if len(self.mapping)==0 or len(words)<50:
            score = self._tfidf_similarity(submitted_text)
            conf = 'high' if score>80 else 'medium' if score>50 else 'low'
            return {
                "overall_score": score,
                "confidence": conf,
                "suspicious_parts": [],
                "message": "TF-IDF fallback used"
            }
        # Semantic matching
        chunks = []
        for i in range(0, max(len(words)-150,1), 150):
            c = words[i:i+200]
            if len(c)>=50: chunks.append(" ".join(c))
        embs = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, num_workers=0)
        faiss.normalize_L2(embs)
        hits, parts = set(), []
        for idx, emb in enumerate(embs):
            D, I = self.index.search(emb.reshape(1,-1), top_k)
            print(f"Chunk {idx} sims: {D[0].tolist()}")
            for score, ref_i in zip(D[0], I[0]):
                if score >= self.threshold:
                    fn, txt = self.mapping[ref_i]
                    parts.append({
                        "submitted_chunk": chunks[idx],
                        "reference_file": fn,
                        "reference_chunk": txt,
                        "similarity": round(float(score),3)
                    })
                    hits.add(idx)
        overall = 100.0 * len(hits) / len(chunks) if chunks else 0.0
        conf = 'high' if overall>80 else 'medium' if overall>50 else 'low'
        return {
            "overall_score": round(overall,1),
            "confidence": conf,
            "suspicious_parts": parts,
            "message": "Semantic match results"
        }
