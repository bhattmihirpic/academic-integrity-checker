from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class PlagiarismDetector:
    """This is our plagiarism detective!"""
    
    def __init__(self):
        """Set up our detective with the right tools"""
        
        # This tool helps us compare texts by looking at important words
        self.text_analyzer = TfidfVectorizer(
            max_features=1000,  # Look at the 1000 most important words
            stop_words='english',  # Ignore common words like "the", "and", "is"
            ngram_range=(1, 3)  # Look at single words, pairs, and triplets
        )
        
        # Some example texts to compare against
        self.reference_texts = [
            "Academic integrity is important in education. Students should do their own work.",
            "The scientific method involves observation, hypothesis, and experimentation.",
            "Climate change is caused by human activities and greenhouse gases.",
            "Technology has changed how we communicate and learn."
        ]
    
    def clean_text_for_analysis(self, text):
        """Clean up text to make it better for comparison"""
        if not text:
            return ""
        
        # Make everything lowercase
        text = text.lower()
        
        # Remove citations like (Smith, 2020)
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def compare_two_texts(self, text1, text2):
        """Compare two pieces of text and give a similarity score"""
        try:
            # Clean up both texts
            clean_text1 = self.clean_text_for_analysis(text1)
            clean_text2 = self.clean_text_for_analysis(text2)
            
            # Use our analyzer to compare them
            texts = [clean_text1, clean_text2]
            text_vectors = self.text_analyzer.fit_transform(texts)
            
            # Calculate similarity (0 = completely different, 1 = identical)
            similarity = cosine_similarity(text_vectors)
            
            # Convert to percentage
            return similarity[0][1] * 100
            
        except Exception as e:
            print(f"Error comparing texts: {e}")
            return 0.0
    
    def find_suspicious_parts(self, text1, text2):
        """Find specific parts that look copied"""
        # Split texts into sentences
        sentences1 = [s.strip() for s in text1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in text2.split('.') if s.strip()]
        
        suspicious_parts = []
        
        # Compare each sentence from text1 with sentences from text2
        for i, sentence1 in enumerate(sentences1):
            if len(sentence1.split()) < 5:  # Skip very short sentences
                continue
                
            for j, sentence2 in enumerate(sentences2):
                if len(sentence2.split()) < 5:
                    continue
                
                similarity = self.compare_two_texts(sentence1, sentence2)
                
                if similarity > 70:  # If sentences are more than 70% similar
                    suspicious_parts.append({
                        'sentence_from_submitted_text': sentence1,
                        'similar_sentence_found': sentence2,
                        'similarity_percentage': round(similarity, 1),
                        'position_in_text': i
                    })
        
        # Sort by most similar first
        suspicious_parts.sort(key=lambda x: x['similarity_percentage'], reverse=True)
        
        return suspicious_parts[:10]  # Return top 10 most suspicious parts
    
    def check_for_plagiarism(self, submitted_text):
        """Check if submitted text looks like it was copied"""
        
        if not submitted_text or len(submitted_text.strip()) < 50:
            return {
                'overall_score': 0.0,
                'confidence': 'low',
                'suspicious_parts': [],
                'message': 'Text too short to analyze properly'
            }
        
        max_similarity = 0.0
        all_suspicious_parts = []
        
        # Compare against all our reference texts
        for reference_text in self.reference_texts:
            similarity = self.compare_two_texts(submitted_text, reference_text)
            
            if similarity > max_similarity:
                max_similarity = similarity
            
            # Find suspicious parts if similarity is high enough
            if similarity > 30:
                parts = self.find_suspicious_parts(submitted_text, reference_text)
                all_suspicious_parts.extend(parts)
        
        # Determine confidence level
        if max_similarity > 80:
            confidence = 'high'
        elif max_similarity > 50:
            confidence = 'medium'  
        else:
            confidence = 'low'
        
        return {
            'overall_score': round(max_similarity, 1),
            'confidence': confidence,
            'suspicious_parts': all_suspicious_parts,
            'message': f'Compared against {len(self.reference_texts)} reference texts'
        }
