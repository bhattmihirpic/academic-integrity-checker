import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class AIDetector:
    """This is our AI content detective!"""
    
    def __init__(self):
        """Set up our AI detective"""
        
        # Try to load a smart AI detection model
        self.ai_model = None
        self.model_working = False
        
        try:
            print("🤖 Loading AI detection model...")
            model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.ai_model = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True
            )
            
            self.model_working = True
            print("✅ AI detection model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Couldn't load AI model: {e}")
            print("Will use pattern-based detection instead")
    
    def look_for_ai_patterns(self, text):
        """Look for patterns that suggest AI wrote this"""
        
        if not text:
            return {'ai_score': 0.0, 'clues_found': []}
        
        text_lower = text.lower()
        clues_found = []
        ai_score = 0.0
        
        # Common phrases AI tools use a lot
        ai_phrases = {
            'as an ai': 30,
            'furthermore': 8, 
            'moreover': 8,
            'in conclusion': 10,
            'it is important to note': 15,
            'however, it is worth noting': 12,
            'on the other hand': 8,
            'additionally': 6,
            'consequently': 8
        }
        
        # Look for these phrases
        for phrase, points in ai_phrases.items():
            count = text_lower.count(phrase)
            if count > 0:
                ai_score += points * count
                clues_found.append(f"Uses '{phrase}' {count} time(s) - AI tools love this phrase!")
        
        # Check if sentences are all very similar length (AI pattern)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 3:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            # Check if all sentences are weirdly similar in length
            if avg_length > 20 and all(abs(length - avg_length) < 5 for length in sentence_lengths):
                ai_score += 15
                clues_found.append(f"All sentences are suspiciously similar length (avg: {avg_length:.1f} words)")
        
        # Check for too many fancy transition words
        transition_words = ['however', 'furthermore', 'moreover', 'consequently', 'therefore', 'nevertheless']
        transition_count = sum(text_lower.count(word) for word in transition_words)
        
        if transition_count > len(sentences) * 0.3:  # More than 30% of sentences have transitions
            ai_score += 12
            clues_found.append(f"Uses too many fancy transition words ({transition_count} found)")
        
        # Convert to percentage (cap at 100)
        ai_percentage = min(100.0, ai_score)
        
        return {
            'ai_score': ai_percentage,
            'clues_found': clues_found
        }
    
    def use_smart_ai_model(self, text):
        """Use the smart AI detection model if available"""
        
        if not self.model_working:
            return None
        
        try:
            # Split text into chunks (the model can only handle so much at once)
            words = text.split()
            chunks = []
            
            # Make chunks of about 400 words each
            for i in range(0, len(words), 400):
                chunk = ' '.join(words[i:i + 400])
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
            
            if not chunks:
                return None
            
            chunk_results = []
            total_ai_score = 0.0
            
            # Analyze each chunk
            for chunk in chunks:
                result = self.ai_model(chunk)
                
                # Extract AI probability from the result
                ai_probability = 0.0
                for item in result:
                    label = item['label'].lower()
                    if 'ai' in label or 'fake' in label or '1' in label:
                        ai_probability = item['score']
                        break
                    elif 'human' in label or 'real' in label or '0' in label:
                        ai_probability = 1.0 - item['score']
                        break
                
                chunk_results.append({
                    'chunk_text': chunk[:100] + '...',
                    'ai_probability': ai_probability * 100
                })
                
                total_ai_score += ai_probability
            
            # Calculate average
            avg_ai_score = (total_ai_score / len(chunks)) * 100
            
            return {
                'ai_score': avg_ai_score,
                'chunk_details': chunk_results,
                'chunks_analyzed': len(chunks)
            }
            
        except Exception as e:
            print(f"Error using AI model: {e}")
            return None
    
    def detect_ai_content(self, text):
        """Main function to detect if AI wrote this text"""
        
        if not text or len(text.strip()) < 50:
            return {
                'ai_probability': 0.0,
                'confidence': 'low',
                'details': 'Text too short to analyze',
                'clues_found': []
            }
        
        # Try the smart model first
        smart_result = self.use_smart_ai_model(text)
        
        # Always do pattern analysis
        pattern_result = self.look_for_ai_patterns(text)
        
        # Combine results
        if smart_result:
            # Use weighted average: 70% smart model, 30% patterns
            final_score = (smart_result['ai_score'] * 0.7) + (pattern_result['ai_score'] * 0.3)
            method_used = "Smart AI model + Pattern analysis"
        else:
            # Use only pattern analysis
            final_score = pattern_result['ai_score']
            method_used = "Pattern analysis only"
        
        # Determine confidence
        if final_score > 80:
            confidence = 'high'
        elif final_score > 60:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'ai_probability': round(final_score, 1),
            'confidence': confidence,
            'details': f'Analysis completed using: {method_used}',
            'clues_found': pattern_result['clues_found'],
            'smart_model_working': self.model_working
        }
