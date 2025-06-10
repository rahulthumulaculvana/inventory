# voice_assistant.py
import logging
import re
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger("VoiceAssistant")

class VoiceAssistant:
    """
    Backend voice assistant that handles voice-specific responses
    Separates display content from speech content
    """
    
    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    
    def process_voice_query(self, user_question: str, rag_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a voice query and generate both display and speech responses
        
        Args:
            user_question (str): The original user question
            rag_response (Dict): Response from RAG system
            
        Returns:
            Dict containing both display_response and speech_response
        """
        try:
            # Extract text content from HTML response
            display_content = rag_response.get("response", "")
            text_content = self._extract_text_from_html(display_content)
            
            # Generate voice-optimized response
            speech_response = self._generate_voice_response(user_question, text_content)
            
            return {
                "display_response": display_content,  # Keep original HTML for display
                "speech_response": speech_response,   # Clean text for speech
                "input_method": "voice"
            }
            
        except Exception as e:
            logger.error(f"Error processing voice query: {str(e)}")
            return {
                "display_response": rag_response.get("response", ""),
                "speech_response": "I encountered an error processing your voice request. Please try again.",
                "input_method": "voice"
            }
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text content from HTML, preserving important structure
        """
        if not html_content:
            return ""
        
        try:
            # Remove HTML tags but preserve some structure
            text = html_content
            
            # Replace headers with line breaks
            text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\1\n', text, flags=re.DOTALL)
            
            # Replace paragraphs with line breaks
            text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n', text, flags=re.DOTALL)
            
            # Replace list items with bullets
            text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', text, flags=re.DOTALL)
            
            # Replace divs with line breaks
            text = re.sub(r'<div[^>]*>(.*?)</div>', r'\1\n', text, flags=re.DOTALL)
            
            # Remove all remaining HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Clean up whitespace
            text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
            text = re.sub(r'\s+', ' ', text)       # Normalize spaces
            text = text.strip()
            
            # Decode HTML entities
            text = text.replace('&nbsp;', ' ')
            text = text.replace('&amp;', '&')
            text = text.replace('&lt;', '<')
            text = text.replace('&gt;', '>')
            text = text.replace('&quot;', '"')
            text = text.replace('&#x27;', "'")
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            # Fallback: just remove all HTML tags
            return re.sub(r'<[^>]+>', '', html_content)
    
    def _generate_voice_response(self, user_question: str, extracted_text: str) -> str:
        """
        Generate a voice-optimized response using AI
        """
        try:
            system_prompt = """You are a helpful restaurant inventory assistant optimized for voice responses.

Convert the given text into a natural, conversational response suitable for text-to-speech. Follow these guidelines:

1. Use conversational language (contractions, natural flow)
2. Keep responses concise but informative 
3. Use "and" instead of bullet points or lists
4. Avoid technical jargon when possible
5. Make numbers and prices easy to understand when spoken
6. End with a helpful follow-up suggestion when appropriate

Make it sound natural when spoken aloud, as if you're having a conversation."""

            user_prompt = f"""Original question: "{user_question}"

Information to convey: {extracted_text}

Convert this into a natural voice response (maximum 200 words):"""

            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            voice_response = response.choices[0].message.content.strip()
            
            # Additional cleanup for voice
            voice_response = self._optimize_for_speech(voice_response)
            
            return voice_response
            
        except Exception as e:
            logger.error(f"Error generating voice response: {str(e)}")
            # Fallback to extracted text
            return self._optimize_for_speech(extracted_text)
    
    def _optimize_for_speech(self, text: str) -> str:
        """
        Final optimizations to make text speech-friendly
        """
        # Replace common symbols with spoken equivalents
        text = text.replace('$', 'dollar ')
        text = text.replace('%', ' percent')
        text = text.replace('&', ' and ')
        text = text.replace('#', ' number ')
        
        # Handle price formatting
        text = re.sub(r'\$(\d+)\.(\d{2,3})', r'$\1 dollars and \2 cents', text)
        text = re.sub(r'\$(\d+)', r'\1 dollars', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[•→▪▫‣⁃]', '', text)  # Remove bullet points
        text = re.sub(r'[^\w\s.,!?-]', '', text)  # Keep only basic punctuation
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def create_greeting_response(self) -> Dict[str, Any]:
        """Create voice-optimized greeting"""
        display = """
        <div style="background-color: #f0f9ff; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <p style="margin: 0; color: #1e40af; font-size: 18px;">👋 Hello!</p>
            <p style="margin: 8px 0 0 0; color: #475569;">I'm your restaurant inventory assistant. I can help you find items, update prices and quantities, and search for market information. What would you like to know?</p>
        </div>
        """
        
        speech = "Hello! I'm your restaurant inventory assistant. I can help you find items, update prices and quantities, and search for market information. What would you like to know?"
        
        return {
            "display_response": display,
            "speech_response": speech,
            "input_method": "voice"
        }
    
    def create_error_response(self, error_message: str = None) -> Dict[str, Any]:
        """Create voice-optimized error response"""
        default_speech = "I'm sorry, I encountered an error processing your request. Could you please try asking your question again?"
        
        display = f"""
        <div style="background-color: #fef2f2; padding: 16px; border-radius: 8px; border-left: 4px solid #ef4444;">
            <p style="margin: 0; color: #dc2626;">❌ Error</p>
            <p style="margin: 8px 0 0 0; color: #7f1d1d;">{error_message or "I encountered an error processing your request. Please try again."}</p>
        </div>
        """
        
        return {
            "display_response": display,
            "speech_response": error_message or default_speech,
            "input_method": "voice"
        }