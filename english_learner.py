#!/usr/bin/env python3
"""
English Learning Helper
A tool to load vocabulary words, generate native speaker sentences, and create fill-in-the-blank tests.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set
import re
import html
import json
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv

# Lazy import for google.generativeai to avoid compatibility issues
genai = None

# Load environment variables
load_dotenv()

console = Console()


class WordTracker:
    """Tracks processed vocabulary words to identify new words and duplicates."""
    
    def __init__(self, tracker_file: str = "processed_words.json"):
        self.tracker_file = tracker_file
        self.processed_words: Set[str] = set()
        self.word_occurrences: Dict[str, int] = {}  # Track how many times each word appears
        self.load()
    
    def load(self) -> None:
        """Load previously processed words from file."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_words = set(data.get('words', []))
                    self.word_occurrences = data.get('occurrences', {})
            except Exception as e:
                console.print(f"[yellow][WARN] Could not load tracker file: {e}[/yellow]")
                self.processed_words = set()
                self.word_occurrences = {}
    
    def save(self) -> None:
        """Save processed words to file."""
        try:
            data = {
                'words': sorted(list(self.processed_words)),
                'occurrences': self.word_occurrences,
                'last_updated': datetime.now().isoformat(),
                'total_words': len(self.processed_words)
            }
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[yellow][WARN] Could not save tracker file: {e}[/yellow]")
    
    def get_new_words(self, words: List[str]) -> tuple[List[str], List[str]]:
        """
        Return new words and duplicate words.
        
        Returns:
            Tuple of (new_words, duplicate_words)
        """
        new_words = []
        duplicate_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.processed_words:
                new_words.append(word)
            else:
                duplicate_words.append(word)
        
        return new_words, duplicate_words
    
    def add_words(self, words: List[str]) -> None:
        """Add words to the processed set and increment occurrence count."""
        for word in words:
            word_lower = word.lower()
            self.processed_words.add(word_lower)
            self.word_occurrences[word_lower] = self.word_occurrences.get(word_lower, 0) + 1
    
    def get_occurrence_count(self, word: str) -> int:
        """Get how many times a word has appeared."""
        word_lower = word.lower()
        # If word is in processed_words but not in occurrences, it was added before tracking
        # So it appeared at least once
        if word_lower in self.processed_words and word_lower not in self.word_occurrences:
            return 1  # Was processed before, so at least 1 occurrence
        return self.word_occurrences.get(word_lower, 0)
    
    def clear(self) -> None:
        """Clear all tracked words."""
        self.processed_words = set()
        self.word_occurrences = {}
        if os.path.exists(self.tracker_file):
            try:
                os.remove(self.tracker_file)
                console.print(f"[green][OK] Cleared tracker file: {self.tracker_file}[/green]")
            except Exception as e:
                console.print(f"[red][ERROR] Could not delete tracker file: {e}[/red]")
    
    def get_stats(self) -> Dict:
        """Get statistics about processed words."""
        duplicates = sum(1 for count in self.word_occurrences.values() if count > 1)
        return {
            'total_processed': len(self.processed_words),
            'total_occurrences': sum(self.word_occurrences.values()),
            'duplicate_words': duplicates,
            'tracker_file': self.tracker_file
        }


class VocabularyLoader:
    """Handles loading and extracting vocabulary words from files."""

    def __init__(self):
        self.words = []
        self.word_pronunciations = {}  # Cache pronunciations
    
    def _get_base_word(self, word: str) -> str:
        """Try to find the base/root word by stripping common suffixes."""
        word_lower = word.lower()
        
        # Common suffixes to try removing (in order of likelihood)
        suffixes = [
            # Past tense and past participle
            ('ied', 'y'), ('ied', 'ie'),  # studied -> study, tried -> try
            ('ed', ''),  # pummeled -> pummel, berated -> berate
            # Present participle and gerund - special cases first
            ('ating', 'ate'),  # ingratiating -> ingratiate, creating -> create
            ('ying', 'y'),  # trying -> try
            ('ing', ''),  # searing -> sear, pummeled -> pummel
            # Plural and third person singular
            ('ies', 'y'),  # cities -> city
            ('ies', 'ie'),  # studies -> study
            ('es', ''),  # boxes -> box, rations -> ration
            # Only strip 's' if word is longer than 5 chars (to avoid stripping from words like "tirade", "constellation")
            # This helps with plurals like "rations" -> "ration" but preserves words ending in 's' naturally
            ('s', ''),  # rations -> ration (but not "tirade" -> "tirad")
        ]
        
        # Try removing suffixes
        for suffix, replacement in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                # Special handling for 's' suffix - only strip if word is longer than 5 chars
                # This avoids stripping 's' from words that naturally end in 's' like "tirade", "constellation"
                if suffix == 's' and len(word_lower) <= 5:
                    continue
                
                base = word_lower[:-len(suffix)] + replacement
                if len(base) > 2:  # Ensure base word is meaningful
                    return base
        
        return word_lower
    
    def _fetch_pronunciation_for_word(self, word: str) -> str:
        """Fetch pronunciation for a single word (helper method).
        
        Uses the base word (not the original word) for pronunciations.
        """
        # Use the same method from SentenceGenerator
        # Import here to avoid circular dependency
        try:
            import urllib.request
            import urllib.parse
            import urllib.error
            import json
            
            # Use base word for pronunciation
            base_word = self._get_base_word(word)
            
            # Try base word first
            pronunciation = self._try_fetch_pronunciation_api(base_word)
            if pronunciation:
                return pronunciation
            
            # If base word failed and it's different from original, try original as fallback
            if base_word != word.lower():
                pronunciation = self._try_fetch_pronunciation_api(word)
                if pronunciation:
                    return pronunciation
            
        except:
            pass
        
        return ""
    
    def _try_fetch_pronunciation_api(self, word: str) -> str:
        """Try to fetch pronunciation from Free Dictionary API."""
        try:
            import urllib.request
            import urllib.parse
            import urllib.error
            import json
            
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word.lower())}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'English-Learning-Helper/1.0')
            
            # Reduced timeout from 10 to 5 seconds for faster failure and better performance
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                if data and len(data) > 0:
                    phonetic = data[0].get('phonetic', '')
                    if phonetic:
                        return phonetic.strip('/')
                    
                    phonetics = data[0].get('phonetics', [])
                    if phonetics and len(phonetics) > 0:
                        for phonetic_entry in phonetics:
                            text = phonetic_entry.get('text', '')
                            if text:
                                return text.strip('/')
        except:
            pass
        
        return ""
    
    def format_word_with_pronunciation(self, word: str) -> str:
        """Format a word with its pronunciation notation."""
        pronunciation = self.word_pronunciations.get(word.lower(), '')
        if pronunciation:
            # Remove leading/trailing slashes if present
            pronunciation = pronunciation.strip('/')
            return f"{word} /{pronunciation}/"
        return word
    
    def get_pronunciation(self, word: str) -> str:
        """Get pronunciation for a word (from cache or fetch if needed)."""
        if word.lower() not in self.word_pronunciations:
            pronunciation = self._fetch_pronunciation_for_word(word)
            if pronunciation:
                self.word_pronunciations[word.lower()] = pronunciation
                return pronunciation
        return self.word_pronunciations.get(word.lower(), "")

    def _extract_most_important_word(self, text: str) -> Optional[str]:
        """
        Extract the most important word from a sentence or phrase.
        The most important word is typically the longest, most complex word.
        
        Args:
            text: Sentence or phrase
            
        Returns:
            The most important word, or None if no suitable word found
        """
        # Clean the text
        text = re.sub(r'[^\w\s-]', ' ', text)  # Remove punctuation except hyphens
        text = html.unescape(text)
        
        # Split into words
        words = re.findall(r'\b[\w-]+\b', text.lower())
        
        if not words:
            return None
        
        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'them', 'we', 'us', 'you', 'he', 'she', 'him', 'her',
            'i', 'me', 'my', 'your', 'his', 'her', 'our', 'their', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'some', 'any', 'no', 'not', 'very', 'just', 'only',
            'also', 'more', 'most', 'much', 'many', 'few', 'little', 'other', 'another', 'such', 'than',
            'then', 'there', 'here', 'up', 'down', 'out', 'off', 'over', 'under', 'above', 'below',
            'about', 'around', 'through', 'during', 'before', 'after', 'while', 'since', 'until',
            'from', 'into', 'onto', 'upon', 'toward', 'towards', 'against', 'among', 'between',
            'within', 'without', 'across', 'along', 'beside', 'besides', 'beyond', 'near', 'far'
        }
        
        # Filter out stop words and short words
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not filtered_words:
            # If all words are stop words, return the longest word
            filtered_words = [w for w in words if len(w) > 1]
        
        if not filtered_words:
            return None
        
        # Return the longest word (most likely to be the vocabulary word)
        # If multiple words have the same length, prefer words with more syllables (longer)
        most_important = max(filtered_words, key=lambda w: (len(w), w.count('-') + 1))
        
        # Clean up the word
        most_important = most_important.strip('.,;:!?()[]{}"\'-')
        
        return most_important if most_important else None

    def _parse_kindle_html(self, content: str) -> List[str]:
        """
        Parse Kindle HTML file to extract vocabulary words from highlights.
        
        Args:
            content: HTML content
            
        Returns:
            List of vocabulary words
        """
        words = []
        
        # Extract all noteText divs (Kindle highlights)
        # Pattern: <div class='noteText'>text</div> or <div class='noteText'>text</h3>
        note_text_pattern = r"<div\s+class=['\"]noteText['\"]>([^<]+)</div>|<div\s+class=['\"]noteText['\"]>([^<]+)</h3>"
        matches = re.findall(note_text_pattern, content, re.IGNORECASE)
        
        for match in matches:
            # match is a tuple, get the non-empty part
            text = match[0] if match[0] else match[1]
            if not text:
                continue
            
            # Clean the text
            text = text.strip()
            text = html.unescape(text)
            
            # Remove leading/trailing punctuation and quotes
            text = re.sub(r'^["\'.,;:!?()\[\]{}]+|["\'.,;:!?()\[\]{}]+$', '', text)
            text = text.strip()
            
            if not text:
                continue
            
            # Determine if it's a single word/phrase or a sentence
            word_count = len(text.split())
            
            if word_count <= 3:
                # Likely a single word or short phrase - extract the main word
                word = self._extract_most_important_word(text)
            else:
                # It's a sentence - extract the most important word
                word = self._extract_most_important_word(text)
            
            if word and len(word) > 2:
                words.append(word)
        
        return words

    def load_file(self, file_path: str, fetch_pronunciations: bool = True) -> List[str]:
        """
        Load vocabulary words from a file.

        Args:
            file_path: Path to the vocabulary file
            fetch_pronunciations: If True, fetch pronunciations for all words. If False, skip pronunciation fetching.

        Returns:
            List of extracted words
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Check if it's a Kindle HTML file
                if file_path.lower().endswith('.html') or file_path.lower().endswith('.htm'):
                    # Check if it's a Kindle notes file
                    if 'noteText' in content or 'notebookFor' in content:
                        self.words = self._parse_kindle_html(content)
                    else:
                        # Regular HTML file - strip tags and extract words
                        clean_content = re.sub(r'<[^>]+>', '', content)
                        clean_content = html.unescape(clean_content)
                        self.words = self._extract_words(clean_content)
                else:
                    # Plain text file
                    self.words = self._extract_words(content)

                console.print(f"[green][OK] Loaded {len(self.words)} vocabulary words from {file_path}[/green]")
                
                # Fetch pronunciations only if requested (skip for check-new to optimize)
                if fetch_pronunciations and len(self.words) > 0:
                    self._fetch_pronunciations_batch(self.words)
                
                return self.words
        except FileNotFoundError:
            console.print(f"[red][ERROR] File not found: {file_path}[/red]")
            return []
        except Exception as e:
            console.print(f"[red][ERROR] Error loading file: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return []
    
    def _fetch_pronunciations_batch(self, words: List[str]) -> None:
        """
        Fetch pronunciations for a batch of words using parallel processing.
        
        Args:
            words: List of words to fetch pronunciations for
        """
        console.print("[dim]Fetching pronunciations for words (parallel processing)...[/dim]")
        
        # Filter words that need pronunciation fetching
        words_to_fetch = [word for word in words if word.lower() not in self.word_pronunciations]
        
        if words_to_fetch:
            fetched_count = [0]  # Use list for thread-safe counter
            fetch_lock = threading.Lock()
            
            def fetch_pronunciation(word: str):
                """Fetch pronunciation for a single word."""
                pronunciation = self._fetch_pronunciation_for_word(word)
                if pronunciation:
                    with fetch_lock:
                        self.word_pronunciations[word.lower()] = pronunciation
                        fetched_count[0] += 1
                        count = fetched_count[0]
                        if count % 20 == 0 or count == len(words_to_fetch):
                            console.print(f"[dim]  Fetched pronunciations for {count}/{len(words_to_fetch)} words...[/dim]")
            
            # Use ThreadPoolExecutor for parallel pronunciation fetching
            # Increased workers for faster I/O-bound API calls (can handle many concurrent requests)
            max_workers = min(50, len(words_to_fetch))  # Increased to 50 for even faster fetching
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(fetch_pronunciation, words_to_fetch)
            
            if fetched_count[0] > 0:
                console.print(f"[green]✓ Found pronunciations for {fetched_count[0]} words[/green]")
        else:
            console.print(f"[green]✓ All pronunciations already cached[/green]")

    def _extract_words(self, content: str) -> List[str]:
        """
        Extract vocabulary words from plain text content.

        Args:
            content: Raw text content

        Returns:
            List of extracted words
        """
        # Split by various delimiters and clean up
        words = re.split(r'[\s\n\r\t,;:.!?()[\]{}"\'-]+', content)

        # Filter out empty strings and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}

        filtered_words = []
        for word in words:
            word = word.strip().lower()
            if len(word) > 2 and word not in stop_words and word.isalpha():
                filtered_words.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in filtered_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        return unique_words


class SentenceGenerator:
    """Generates native speaker sentences using Google AI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.use_mock = False
        self.model = None
        self.models = []  # Store multiple models to try

        if not self.api_key:
            console.print("[yellow][WARN] No Google API key found. Using mock sentence generation.[/yellow]")
            console.print("[dim]To use Google AI (Gemini) for realistic sentence generation:[/dim]")
            console.print("[dim]1. Get a free API key from: https://makersuite.google.com/app/apikey[/dim]")
            console.print("[dim]2. Create a .env file in the project directory with: GOOGLE_API_KEY=your_key_here[/dim]")
            self.use_mock = True
        else:
            try:
                # Lazy import to avoid compatibility issues with Python 3.14
                global genai
                if genai is None:
                    # Suppress deprecation warnings
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        try:
                            import google.generativeai as genai
                        except TypeError as te:
                            # Python 3.14 metaclass issue
                            error_msg = str(te)
                            if "Metaclasses with custom tp_new" in error_msg or "metaclass" in error_msg.lower():
                                raise ImportError("Python 3.14 compatibility issue with google-generativeai. Please use Python 3.11 or 3.12, or use mock mode.")
                            raise
                        except Exception as import_error:
                            # Catch any other import errors
                            error_msg = str(import_error)
                            if "Metaclasses with custom tp_new" in error_msg or "metaclass" in error_msg.lower():
                                raise ImportError("Python 3.14 compatibility issue with google-generativeai. Please use Python 3.11 or 3.12, or use mock mode.")
                            raise
                
                # Suppress deprecation warnings during configuration
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    genai.configure(api_key=self.api_key)
                
                # Use gemini-pro as it's the most stable and widely supported model
                # gemini-1.5-flash and gemini-1.5-pro may have API version compatibility issues
                model_names = ['gemini-pro']
                
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    for model_name in model_names:
                        try:
                            model = genai.GenerativeModel(model_name)
                            self.models.append(model)
                            if self.model is None:
                                self.model = model
                                console.print(f"[green][OK] Using Google AI model: {model_name}[/green]")
                        except Exception as model_error:
                            # Check for metaclass error specifically
                            error_msg = str(model_error)
                            if "Metaclasses with custom tp_new" in error_msg or "metaclass" in error_msg.lower():
                                raise ImportError("Python 3.14 compatibility issue with google-generativeai. Please use Python 3.11 or 3.12, or use mock mode.")
                            # Continue trying other models
                            continue
                
                if self.model is None or len(self.models) == 0:
                    raise Exception("Could not initialize any Google AI model")
                
                if len(self.models) > 1:
                    console.print(f"[dim]Initialized {len(self.models)} models for fallback[/dim]")
                    
            except ImportError as ie:
                console.print(f"[yellow][WARN] Import error: {ie}[/yellow]")
                console.print("[yellow]Using mock sentence generation instead.[/yellow]")
                console.print("[dim]Tip: For Python 3.14, consider using Python 3.11 or 3.12 for full Google AI support.[/dim]")
                self.use_mock = True
            except Exception as e:
                error_msg = str(e)
                if "Metaclasses with custom tp_new" in error_msg or "metaclass" in error_msg.lower():
                    console.print(f"[yellow][WARN] Python 3.14 compatibility issue with google-generativeai[/yellow]")
                    console.print("[yellow]Using mock sentence generation instead.[/yellow]")
                    console.print("[dim]Tip: Consider using Python 3.11 or 3.12 for full Google AI support.[/dim]")
                    console.print(f"[dim]Error details: {error_msg}[/dim]")
                else:
                    console.print(f"[yellow][WARN] Failed to initialize Google AI: {e}[/yellow]")
                    console.print("[yellow]Using mock sentence generation instead.[/yellow]")
                self.use_mock = True

    def _try_generate_content(self, prompt: str) -> str:
        """Try to generate content using available models, retrying with alternatives on 404 errors."""
        if not self.models:
            raise Exception("No models available")
        
        last_error = None
        models_tried = []
        
        for idx, model in enumerate(self.models):
            try:
                response = model.generate_content(prompt)
                # Handle different response structures
                text = None
                if response:
                    # Try to get text from response
                    if hasattr(response, 'text') and response.text:
                        text = response.text
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Sometimes response structure is different
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            text = ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        elif hasattr(candidate, 'text'):
                            text = candidate.text
                
                if text:
                    # If this model worked but wasn't the primary, switch to it
                    if model != self.model:
                        self.model = model
                        # Also update models list to prioritize this working model
                        self.models.insert(0, self.models.pop(idx))
                    return text
                else:
                    # Response exists but no text - this is unusual
                    raise Exception(f"Google AI returned response but no text content. Response: {type(response)}")
            except Exception as e:
                error_msg = str(e)
                last_error = e
                models_tried.append(f"model_{idx}")
                
                # Check if it's a 404, model not found, or API version error
                is_404_error = (
                    "404" in error_msg or 
                    "not found" in error_msg.lower() or 
                    "not supported" in error_msg.lower() or
                    "v1beta" in error_msg.lower() or
                    "api version" in error_msg.lower()
                )
                
                # If it's a 404/model error and we have more models to try, continue
                if is_404_error and idx < len(self.models) - 1:
                    continue
                # If it's not a 404 error, raise immediately
                if not is_404_error:
                    raise
                # If it's the last model and it's a 404, we'll raise after the loop
        
        # If all models failed with 404, raise the last error
        if last_error:
            error_msg = str(last_error)
            # Provide a more helpful error message
            if "v1beta" in error_msg.lower():
                raise Exception(f"All models failed. The API may require a different model name or API version. Last error: {error_msg}")
            raise last_error
        raise Exception("Failed to generate content with any available model")

    def generate_explanation_and_examples(self, word: str, num_examples: int = 1, is_repeat: bool = False) -> Dict[str, str]:
        """
        Generate explanation and native speaker examples for a vocabulary word.

        Args:
            word: Vocabulary word
            num_examples: Number of example sentences to generate
            is_repeat: If True, this word appeared before - generate different examples

        Returns:
            Dictionary with 'explanation' and 'examples' keys
        """
        debug_enabled = os.getenv('DEBUG_GOOGLE_AI', '0') == '1'
        
        if self.use_mock:
            if debug_enabled:
                console.print(f"[yellow][DEBUG] Using MOCK mode for '{word}' (Google AI not available)[/yellow]")
            return self._generate_mock_explanation_and_examples(word, num_examples, is_repeat)
        else:
            if debug_enabled:
                console.print(f"[cyan][DEBUG] Using Google AI for '{word}'[/cyan]")
            return self._generate_ai_explanation_and_examples(word, num_examples, is_repeat)

    def generate_sentences(self, words: List[str], num_sentences: int = 3) -> Dict[str, List[str]]:
        """
        Generate native speaker sentences for vocabulary words.

        Args:
            words: List of vocabulary words
            num_sentences: Number of sentences to generate per word

        Returns:
            Dictionary mapping words to lists of generated sentences
        """
        results = {}

        total_words = len(words)
        console.print(f"[bold]Processing {total_words} vocabulary words...[/bold]")

        for idx, word in enumerate(words, 1):
            try:
                if self.use_mock:
                    sentences = self._generate_mock_sentences(word, num_sentences)
                else:
                    sentences = self._generate_ai_sentences(word, num_sentences)

                results[word] = sentences
                if idx % 10 == 0 or idx == total_words:
                    console.print(f"[green][OK] Generated sentences for {idx}/{total_words} words[/green]")

            except Exception as e:
                console.print(f"[yellow][WARN] Failed to generate sentences for '{word}': {e}[/yellow]")
                results[word] = []

        return results

    def _generate_ai_sentences(self, word: str, num_sentences: int) -> List[str]:
        """Generate sentences using Google AI."""
        prompt = f"""
        Generate {num_sentences} natural English sentences that native speakers would use,
        incorporating the vocabulary word "{word}" in context.

        Requirements:
        - Each sentence should demonstrate proper usage of "{word}"
        - Sentences should be natural and conversational
        - Vary the sentence structures (questions, statements, etc.)
        - Keep sentences between 8-15 words

        Format your response as a numbered list:
        1. [Sentence 1]
        2. [Sentence 2]
        3. [Sentence 3]
        """

        text = self._try_generate_content(prompt)
        return self._parse_sentences(text)

    def _generate_ai_explanation_and_examples(self, word: str, num_examples: int, is_repeat: bool = False) -> Dict[str, str]:
        """Generate explanation and examples using Google AI."""
        # Debug: Check if we're actually using Google AI
        debug_enabled = os.getenv('DEBUG_GOOGLE_AI', '0') == '1'
        if debug_enabled:
            console.print(f"[cyan][DEBUG] Calling Google AI for '{word}'...[/cyan]")
        
        repeat_note = ""
        if is_repeat:
            repeat_note = "\n        IMPORTANT: This word appeared in previous learning sessions. Provide a DIFFERENT example sentence than what might have been used before. Use a completely different context or situation to reinforce learning."
        
        prompt = f"""
        Provide a dictionary definition (like Oxford or Google Dictionary) and {num_examples} example sentence(s) for the vocabulary word "{word}".

        Requirements:
        - The EXPLANATION should be a proper dictionary definition in the style of Oxford or Google Dictionary
        - The definition should be clear, precise, and explain what the word means
        - Use the format: "definition text" (like "a fixed amount of a commodity officially allowed to each person during a time of shortage, as in wartime.")
        
        - The EXAMPLE must be a REALISTIC example of how US English native speakers actually use this word in daily conversations
        - The example should sound like actual dialogue from TV shows (like Friends, The Office, Breaking Bad), movies, or real everyday conversations
        - Make it sound natural and conversational - like something a real person would say in casual conversation
        - Use contractions, casual language, and natural speech patterns when appropriate
        - The example should demonstrate the word being used naturally in context, not forced or academic
        - Think about how people actually talk in real life - not formal or textbook-like
        - The sentence should sound like something you'd overhear in a coffee shop, restaurant, or between friends{repeat_note}

        Format your response exactly as follows:

        EXPLANATION:
        [Dictionary definition of "{word}"]

        EXAMPLES:
        1. [One realistic conversational example showing how native speakers actually use "{word}" in daily life - make it sound like real conversation]
        """

        try:
            # Try to generate content with Google AI
            text = self._try_generate_content(prompt)
            
            # Debug: Log if we got a response (but don't spam the console)
            if not text or len(text.strip()) == 0:
                # Google AI returned empty response - this is a real failure
                raise Exception("Google AI returned empty response")
            
            # Parse explanation and examples
            explanation = ""
            examples = []
            
            lines = text.split('\n')
            in_explanation = False
            in_examples = False
            
            for line in lines:
                line = line.strip()
                line_upper = line.upper()
                
                # Check for explanation section header (more flexible matching)
                if 'EXPLANATION:' in line_upper or 'DEFINITION:' in line_upper:
                    in_explanation = True
                    in_examples = False
                    # Extract explanation from the same line if present
                    explanation_text = re.sub(r'^(EXPLANATION|DEFINITION)[:\s]*', '', line, flags=re.IGNORECASE).strip()
                    if explanation_text:
                        explanation = explanation_text
                    continue
                elif 'EXAMPLES:' in line_upper or 'EXAMPLE:' in line_upper:
                    in_explanation = False
                    in_examples = True
                    continue
                
                if in_explanation and line:
                    # Skip separators
                    if line in ['---', '---', '='] or not line:
                        continue
                    # Skip lines that are clearly examples (start with numbers or bullets)
                    if re.match(r'^\d+[\.\)]\s+', line) or line.startswith('•') or line.startswith('-'):
                        # This might be an example that got mixed in, switch to examples
                        in_explanation = False
                        in_examples = True
                        continue
                    if explanation:
                        explanation += " " + line
                    else:
                        explanation = line
                elif in_examples and line:
                    # Remove numbering if present
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if line and line not in ['---', '---', '=']:
                        examples.append(line)
            
            # If explanation is still empty or too short, try to extract from the raw text
            # Sometimes Google AI doesn't follow the exact format
            if not explanation or len(explanation) < 10:
                # Try to find definition-like text in the response
                # Look for sentences that define the word
                text_lower = text.lower()
                word_lower = word.lower()
                
                # Look for patterns like "word is..." or "word means..." or "word refers to..."
                definition_patterns = [
                    rf'{re.escape(word_lower)}\s+(?:is|means|refers to|denotes|describes)\s+([^\.]+(?:\.[^\.]+)*)',
                    rf'{re.escape(word_lower)}\s*:\s*([^\.]+(?:\.[^\.]+)*)',
                ]
                
                for pattern in definition_patterns:
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        # Extract the definition part
                        if match.groups():
                            potential_def = match.group(1).strip()
                        else:
                            # Extract from match position
                            start = match.start()
                            end = min(start + 200, len(text))
                            potential_def = text[start:end].strip()
                            # Clean up
                            potential_def = re.sub(r'^(?:explanation|definition)[:\s]*', '', potential_def, flags=re.IGNORECASE)
                            potential_def = potential_def.split('\n')[0].strip()
                        
                        if len(potential_def) > 20:  # Must be substantial
                            explanation = potential_def
                            break
                    if explanation:
                        break
                
                # If still no explanation, try a more aggressive approach - just take the first substantial sentence
                if not explanation or len(explanation) < 10:
                    # Split text into sentences and find the first one that looks like a definition
                    sentences = re.split(r'[.!?]\s+', text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        # Look for sentences that contain the word and are definition-like
                        if word.lower() in sentence.lower() and len(sentence) > 30:
                            # Check if it looks like a definition (contains definition words)
                            if any(def_word in sentence.lower() for def_word in ['is', 'means', 'refers', 'denotes', 'describes', 'a ', 'an ', 'the ']):
                                explanation = sentence
                                break
                
                # Last resort: if we have text but no explanation, use first paragraph
                if not explanation or len(explanation) < 10:
                    # Take first paragraph (up to first double newline or 200 chars)
                    first_para = text.split('\n\n')[0].strip() if '\n\n' in text else text[:200].strip()
                    if len(first_para) > 20 and word.lower() in first_para.lower():
                        explanation = first_para
                
                # Even more aggressive: if we have text but no explanation, just take first non-empty line
                if not explanation or len(explanation) < 10:
                    # Get all non-empty lines
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    for line in lines:
                        # Skip headers and examples
                        if line.upper() in ['EXPLANATION:', 'DEFINITION:', 'EXAMPLES:', 'EXAMPLE:']:
                            continue
                        # Skip numbered examples
                        if re.match(r'^\d+[\.\)]\s+', line):
                            continue
                        # Take first substantial line that contains the word or looks like a definition
                        if len(line) > 20:
                            if word.lower() in line.lower() or any(def_word in line.lower() for def_word in ['is', 'means', 'refers', 'denotes', 'describes']):
                                explanation = line
                                break
                    # If still nothing, just take the first substantial line
                    if not explanation or len(explanation) < 10:
                        for line in lines:
                            if len(line) > 20:
                                explanation = line
                                break
            
            # Only use API fallback if Google AI completely failed to provide a definition
            # IMPORTANT: Don't override Google AI's definition - it should always be used if present
            # Google AI definitions can be valid even if they're short or start with parentheses
            # Be more lenient - accept any explanation that's not empty and not the generic fallback
            google_ai_provided = bool(explanation and 
                                     len(explanation.strip()) > 0 and 
                                     explanation.strip() != f"{word} is an important vocabulary word used in various contexts." and
                                     'Unable to fetch definition' not in explanation)
            
            # Debug: If parsing failed, log the raw text to help diagnose
            # Enable verbose debug by setting DEBUG_GOOGLE_AI=1 in environment
            debug_enabled = os.getenv('DEBUG_GOOGLE_AI', '0') == '1'
            if not google_ai_provided and text:
                if not hasattr(self, '_parse_failures_logged'):
                    self._parse_failures_logged = set()
                
                # Show debug message if enabled or for first few failures
                should_log = debug_enabled or (word not in self._parse_failures_logged and len(self._parse_failures_logged) < 2)
                if should_log:
                    # Log parsing failures to help debug - make it more visible
                    console.print(f"[bold yellow][DEBUG] Google AI parsing failed for '{word}'[/bold yellow]")
                    console.print(f"[yellow]Raw response length: {len(text)} chars[/yellow]")
                    console.print(f"[yellow]Raw response (first 500 chars):[/yellow]")
                    console.print(f"[dim]{text[:500]}[/dim]")
                    console.print(f"[yellow]Extracted explanation: '{explanation[:100] if explanation else '(empty)'}'[/yellow]")
                    console.print(f"[yellow]Explanation length: {len(explanation) if explanation else 0} chars[/yellow]")
                    if not debug_enabled:
                        self._parse_failures_logged.add(word)
            
            # Only try API fallback if Google AI didn't provide a valid definition
            # Note: This fallback happens inside generate_explanation_and_examples, so it won't show progress
            # The progress message will be shown in process_single_word if this fallback also fails
            if not google_ai_provided:
                # Try fetching from online API as last resort (without progress callback here)
                # Progress will be shown in process_single_word if needed
                api_definition = self._fetch_definition_from_api(word)
                if api_definition and 'Unable to fetch definition' not in api_definition:
                    explanation = api_definition
            
            # Final fallback - should rarely be needed
            # Only use generic message if we truly have no definition from either source
            if not explanation or (len(explanation) < 10 and not google_ai_provided):
                explanation = f"{word} is an important vocabulary word used in various contexts."
            
            # Fetch pronunciation from online API
            pronunciation = self._fetch_pronunciation_from_api(word)
            
            return {
                'explanation': explanation,
                'examples': examples[:num_examples] if examples else self._generate_fallback_examples(word, num_examples, is_repeat),
                'pronunciation': pronunciation
            }
        except Exception as e:
            error_msg = str(e)
            # Only show warning if it's not a 404/model not found error (those are expected and handled silently)
            if "404" not in error_msg and "not found" not in error_msg.lower() and "not supported" not in error_msg.lower():
                console.print(f"[yellow][WARN] AI generation failed for '{word}': {e}[/yellow]")
            # Fallback to online API-based generation
            return self._generate_mock_explanation_and_examples(word, num_examples, is_repeat)

    def _generate_mock_explanation_and_examples(self, word: str, num_examples: int, is_repeat: bool = False) -> Dict[str, str]:
        """Generate mock explanation and examples for demonstration purposes."""
        # Always try to fetch definition from online APIs first
        explanation = self._fetch_definition_from_api(word)
        
        # Only use local fallback if all APIs fail (minimal local dictionary)
        if not explanation:
            # Minimal fallback dictionary - only for very common words as last resort
            minimal_fallback = {
                'the': 'definite article',
                'a': 'indefinite article',
                'an': 'indefinite article',
            }
            if word.lower() in minimal_fallback:
                explanation = minimal_fallback[word.lower()]
            else:
                # Use original word only (no base word fallback for explanations)
                explanation = f"Unable to fetch definition for '{word}'. The word may not be in the dictionary, or there may be a network issue. Please check your internet connection or set up Google API key for better results."
        
        # Conversational examples that sound like they come from TV shows or daily conversations
        conversational_examples = {
            'rations': [
                "During the war, they had to survive on meager rations.",
                "I'm running low on food - these rations won't last the week.",
                "The soldiers were given daily rations of bread and water.",
            ],
            'searing': [
                "That pan is searing hot - be careful!",
                "I felt a searing pain in my back after lifting that box.",
                "The sun was searing down on us during the hike.",
            ],
            'ingratiating': [
                "He was being so ingratiating, trying to get on her good side.",
                "Stop being so ingratiating - just be yourself.",
                "She gave him an ingratiating smile, hoping he'd help her out.",
            ],
            'pummeled': [
                "The boxer pummeled his opponent until the referee stopped the fight.",
                "I got pummeled by my boss in that meeting - it was brutal.",
                "The rain pummeled against the windows all night.",
            ],
            'berated': [
                "My mom berated me for coming home late again.",
                "The coach berated the team after their terrible performance.",
                "She berated him for forgetting their anniversary.",
            ],
            'tirade': [
                "He went on a tirade about how terrible the service was.",
                "My boss launched into a tirade about missed deadlines.",
                "She unleashed a tirade of complaints about the new policy.",
            ],
            'unhinged': [
                "He's completely unhinged - I wouldn't trust him with anything.",
                "That guy went totally unhinged when he found out.",
                "She seemed a bit unhinged after working three night shifts in a row.",
            ],
            'perpetrator': [
                "The police are still looking for the perpetrator of the crime.",
                "They caught the perpetrator red-handed.",
                "We need to find out who the real perpetrator is.",
            ],
            'fabulist': [
                "Don't believe him - he's a known fabulist.",
                "She's such a fabulist, always making up stories.",
                "That guy is a complete fabulist - nothing he says is true.",
            ],
            'concedes': [
                "Fine, I concede - you were right about this one.",
                "He finally concedes that maybe he was wrong.",
                "She concedes the point but still doesn't agree with the plan.",
            ],
            'autocracy': [
                "That company runs like an autocracy - one person makes all the decisions.",
                "We don't want an autocracy here - everyone should have a say.",
                "The country moved from democracy to autocracy under his rule.",
            ],
            'stern': [
                "My dad gave me a stern look when I came home late.",
                "She was very stern with the kids about not running in the house.",
                "The teacher's stern warning stopped all the talking.",
            ],
            'adversity': [
                "She's faced a lot of adversity in her life, but she never gives up.",
                "We need to learn how to deal with adversity.",
                "Despite all the adversity, they managed to succeed.",
            ],
            'malady': [
                "What malady is keeping you from work today?",
                "The doctor couldn't identify the exact malady.",
                "This mysterious malady has been affecting people for weeks.",
            ],
            'aversion': [
                "I have a strong aversion to seafood - I can't stand it.",
                "She has an aversion to public speaking.",
                "His aversion to conflict makes him avoid difficult conversations.",
            ],
            'siege': [
                "The city was under siege for months.",
                "They laid siege to the castle.",
                "We're under siege from all these complaints.",
            ],
            'compulsion': [
                "I have this compulsion to check my phone every five minutes.",
                "She felt a compulsion to tell him the truth.",
                "There's no compulsion - you can do it if you want.",
            ],
            'summarily': [
                "He was summarily dismissed from his job.",
                "The case was summarily dismissed by the judge.",
                "They summarily rejected our proposal without even looking at it.",
            ],
            'ruefully': [
                "He smiled ruefully and admitted he'd made a mistake.",
                "She looked at him ruefully, knowing she'd hurt his feelings.",
                "I ruefully realized I'd left my keys at home.",
            ],
            'wistful': [
                "She had a wistful look in her eyes when she talked about her childhood.",
                "He spoke wistfully about the good old days.",
                "There was something wistful about the way she said goodbye.",
            ],
            'constellation': [
                "We could see the Big Dipper constellation clearly last night.",
                "Orion is my favorite constellation - it's so easy to spot.",
                "She pointed out the constellation of Cassiopeia in the night sky.",
            ],
        }
        
        # Alternate examples for repeat words (different contexts)
        repeat_examples = {
            'rations': [
                "We need to ration our supplies until help arrives.",
                "The camp had strict rationing rules during the emergency.",
            ],
            'searing': [
                "The memory of that moment is seared into my mind forever.",
                "She gave him a searing look that made him stop talking.",
            ],
            'berated': [
                "He berated himself for making such a stupid mistake.",
                "The customer berated the waiter over a small error.",
            ],
            'tirade': [
                "She went on a tirade against social media influencers.",
                "His tirade lasted for twenty minutes before he calmed down.",
            ],
        }
        
        # Use conversational example if available
        if word.lower() in conversational_examples:
            import random
            if is_repeat and word.lower() in repeat_examples:
                # Use different examples for repeats
                examples = random.sample(repeat_examples[word.lower()], min(num_examples, len(repeat_examples[word.lower()])))
            else:
                # Use regular examples
                examples = random.sample(conversational_examples[word.lower()], min(num_examples, len(conversational_examples[word.lower()])))
        else:
            # Generate a more conversational example
            if is_repeat:
                # Different templates for repeats
                conversational_templates = [
                    f"Let me give you another example of {word} - this time in a different context.",
                    f"I've seen {word} used differently - like in this situation.",
                    f"Here's another way people use {word} in everyday conversation.",
                    f"Another example of {word} would be in this scenario.",
                ]
            else:
                conversational_templates = [
                    f"I can't believe how {word} that situation was.",
                    f"She's always talking about {word} - it's annoying.",
                    f"Did you see how {word} he got?",
                    f"That's so {word} - I've never seen anything like it.",
                    f"He's been {word} about this for weeks now.",
                ]
            import random
            random.shuffle(conversational_templates)
            examples = conversational_templates[:num_examples]
        
        # Fetch pronunciation from online API
        pronunciation = self._fetch_pronunciation_from_api(word)
        
        return {
            'explanation': explanation,
            'examples': examples,
            'pronunciation': pronunciation
        }
    
    def _generate_fallback_examples(self, word: str, num_examples: int, is_repeat: bool = False) -> List[str]:
        """Generate simple fallback examples when AI generation fails."""
        import random
        if is_repeat:
            templates = [
                f"Here's another example of {word} in a different context.",
                f"I've seen {word} used differently - like in this situation.",
                f"Another way people use {word} in everyday conversation.",
            ]
        else:
            templates = [
                f"I can't believe how {word} that situation was.",
                f"She's always talking about {word} - it's interesting.",
                f"Did you see how {word} he got?",
                f"That's so {word} - I've never seen anything like it.",
            ]
        random.shuffle(templates)
        return templates[:num_examples]
    
    def _fetch_definition_from_api(self, word: str, progress_callback: Optional[callable] = None) -> str:
        """Fetch definition from online dictionary APIs (Free Dictionary API, Oxford, etc.).
        
        Uses the word itself (not the base word) for explanations.
        
        Args:
            word: Word to fetch definition for
            progress_callback: Optional callback to report progress with source info
        """
        # Use the word as-is (no base word fallback for explanations)
        definition, source = self._try_fetch_definition_apis(word, progress_callback)
        return definition if definition else ""
    
    def _try_fetch_oxford_definition(self, word: str, app_id: str, api_key: str) -> str:
        """Fetch definition from Oxford Languages (Oxford Dictionary API)."""
        import urllib.request
        import urllib.parse
        import urllib.error
        import json
        
        try:
            url = f"https://od-api.oxforddictionaries.com/api/v2/entries/en-gb/{urllib.parse.quote(word.lower())}"
            req = urllib.request.Request(url)
            req.add_header('app_id', app_id)
            req.add_header('app_key', api_key)
            req.add_header('User-Agent', 'English-Learning-Helper/1.0')
            
            # Reduced timeout from 10 to 5 seconds for faster failure and better performance
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                if 'results' in data and len(data['results']) > 0:
                    lexical_entries = data['results'][0].get('lexicalEntries', [])
                    if lexical_entries:
                        entries = lexical_entries[0].get('entries', [])
                        if entries:
                            senses = entries[0].get('senses', [])
                            if senses:
                                definitions = senses[0].get('definitions', [])
                                if definitions:
                                    definition = definitions[0].strip()
                                    if definition:
                                        return definition
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
            pass
        except Exception as e:
            pass
        
        return ""
    
    def _try_fetch_google_definition(self, word: str) -> str:
        """Fetch definition from Google Dictionary via web scraping.
        
        Note: Google's HTML structure changes frequently and may require JavaScript rendering.
        This method attempts to extract definitions but may not always succeed.
        For more reliable results, consider using Oxford Dictionary API or Free Dictionary API.
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        import re
        
        try:
            # Use Google's "define:" search feature
            search_url = f"https://www.google.com/search?q=define%3A{urllib.parse.quote(word.lower())}"
            req = urllib.request.Request(search_url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
            req.add_header('Accept-Language', 'en-US,en;q=0.9')
            req.add_header('Accept-Encoding', 'gzip, deflate')
            
            # Reduced timeout from 10 to 5 seconds for faster failure and better performance
            with urllib.request.urlopen(req, timeout=5) as response:
                # Handle gzip encoding if present
                html = response.read()
                try:
                    import gzip
                    html = gzip.decompress(html)
                except:
                    pass
                html = html.decode('utf-8', errors='ignore')
                
                # Try BeautifulSoup if available for better parsing
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for definition in various Google dictionary card structures
                    # Try multiple selectors that Google uses
                    selectors = [
                        'div[data-dobid="dfn"]',
                        'div.LGOjhe',
                        'div.LTKOO',
                        'div.sY7ric',
                        'div.kno-fv',
                        'div.BNeawe',
                        'span.BNeawe',
                    ]
                    
                    for selector in selectors:
                        elements = soup.select(selector)
                        for elem in elements:
                            text = elem.get_text(strip=True)
                            if text and 20 < len(text) < 500:
                                # Check if it looks like a definition
                                text_lower = text.lower()
                                if any(indicator in text_lower for indicator in ['is', 'are', 'was', 'were', 'means', 'refers', 'denotes', 'a ', 'an ', 'the ', 'of ', 'in ', 'on ']):
                                    # Avoid common non-definition text
                                    if not any(skip in text_lower for skip in ['see also', 'related', 'synonyms', 'antonyms', 'examples', 'usage']):
                                        return text.strip()
                    
                    # Try to find definition in data attributes
                    dfn_divs = soup.find_all('div', {'data-dobid': 'dfn'})
                    for div in dfn_divs:
                        text = div.get_text(strip=True)
                        if text and 20 < len(text) < 500:
                            return text.strip()
                            
                except ImportError:
                    # BeautifulSoup not available, use regex fallback
                    pass
                
                # Regex fallback patterns
                # Pattern 1: Look for definition in structured data (JSON-LD) - most reliable
                pattern1 = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
                matches = re.findall(pattern1, html, re.DOTALL | re.IGNORECASE)
                for json_ld in matches:
                    try:
                        import json
                        data = json.loads(json_ld)
                        if isinstance(data, dict):
                            # Look for description or definition
                            description = (data.get('description', '') or 
                                         data.get('definition', '') or 
                                         data.get('detailedDescription', {}).get('articleBody', '') or
                                         data.get('description', ''))
                            if description and len(description) > 20:
                                return description.strip()
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    description = (item.get('description', '') or 
                                                 item.get('definition', '') or
                                                 item.get('detailedDescription', {}).get('articleBody', ''))
                                    if description and len(description) > 20:
                                        return description.strip()
                    except:
                        pass
                
                # Pattern 2: Look for "data-dobid" attribute with definition
                pattern2 = r'<div[^>]*data-dobid=["\']dfn["\'][^>]*>.*?<span[^>]*>([^<]{20,300})</span>'
                match = re.search(pattern2, html, re.IGNORECASE | re.DOTALL)
                if match:
                    definition = match.group(1).strip()
                    if definition and len(definition) > 20:
                        return definition
                        
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            pass
        except Exception as e:
            pass
        
        return ""
    
    def _try_fetch_definition_apis(self, word: str, progress_callback: Optional[callable] = None) -> tuple:
        """Try to fetch definition from all available APIs.
        
        Optimized priority order for speed and accuracy:
        1. Free Dictionary API (dictionaryapi.dev) - FASTEST, free, reliable, no API key needed
        2. Oxford Languages (Oxford Dictionary API) - MOST ACCURATE, requires API keys
        3. WordsAPI (if API key available) - Alternative source
        
        Note: Google scraping removed due to unreliability and slowness.
        
        Returns:
            Tuple of (definition, source_name) where source_name is the API that provided the definition
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        import json
        import time
        
        # Minimal delay - parallel processing handles rate limiting better
        # Reduced delay since we're using many parallel threads
        time.sleep(0.05)  # Very small delay for rate limit protection
        
        # 1. Free Dictionary API (dictionaryapi.dev) - FASTEST and most reliable free option
        # This is prioritized first because it's fast, free, and doesn't require API keys
        # Try up to 2 times to handle transient network issues
        for attempt in range(2):
            try:
                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word.lower())}"
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'English-Learning-Helper/1.0')
                req.add_header('Accept', 'application/json')
                
                # Reduced timeout for faster failure if API is slow
                with urllib.request.urlopen(req, timeout=8) as response:
                    response_data = response.read().decode()
                    data = json.loads(response_data)
                    
                    if data and len(data) > 0:
                        # Get the first meaning - try multiple paths
                        meanings = data[0].get('meanings', [])
                        if meanings and len(meanings) > 0:
                            # Try first meaning's definitions
                            definitions = meanings[0].get('definitions', [])
                            if definitions and len(definitions) > 0:
                                definition = definitions[0].get('definition', '').strip()
                                if definition:
                                    if progress_callback:
                                        progress_callback(f"Getting definition for '{word}'... (from Free Dictionary API)")
                                    return definition, "Free Dictionary API"
                            
                            # If no definition in first meaning, try other meanings
                            for meaning in meanings[1:]:
                                definitions = meaning.get('definitions', [])
                                if definitions and len(definitions) > 0:
                                    definition = definitions[0].get('definition', '').strip()
                                    if definition:
                                        if progress_callback:
                                            progress_callback(f"Getting definition for '{word}'... (from Free Dictionary API)")
                                        return definition, "Free Dictionary API"
                    # If we got data but no definition, break (don't retry)
                    break
            except urllib.error.HTTPError as e:
                # 404 means word not found - don't retry
                if e.code == 404:
                    break
                # 429 means rate limited - wait longer before retry
                if e.code == 429:
                    if attempt == 0:
                        time.sleep(2.0)  # Wait 2 seconds for rate limit
                        continue
                    break
                # Other HTTP errors - retry once with delay
                if attempt == 0:
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                break
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
                # Network or parsing errors - retry once
                if attempt == 0:
                    import time
                    time.sleep(0.5)
                    continue
                break
            except Exception as e:
                # Other errors - don't retry
                break
        
        # 2. Oxford Languages (Oxford Dictionary API) - MOST ACCURATE, requires API keys
        # Only try if Free Dictionary API failed
        oxford_app_id = os.getenv('OXFORD_APP_ID')
        oxford_api_key = os.getenv('OXFORD_API_KEY')
        if oxford_app_id and oxford_api_key:
            definition = self._try_fetch_oxford_definition(word, oxford_app_id, oxford_api_key)
            if definition:
                if progress_callback:
                    progress_callback(f"Getting definition for '{word}'... (from Oxford Dictionary API)")
                return definition, "Oxford Dictionary API"
        
        # 3. WordsAPI (if API key is available) - Alternative source
        words_api_key = os.getenv('WORDS_API_KEY')
        if words_api_key:
            try:
                url = f"https://wordsapiv1.p.rapidapi.com/words/{urllib.parse.quote(word.lower())}/definitions"
                req = urllib.request.Request(url)
                req.add_header('X-RapidAPI-Key', words_api_key)
                req.add_header('X-RapidAPI-Host', 'wordsapiv1.p.rapidapi.com')
                req.add_header('User-Agent', 'English-Learning-Helper/1.0')
                
                # Reduced timeout from 10 to 5 seconds for faster failure and better performance
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    
                    if 'definitions' in data and len(data['definitions']) > 0:
                        definition = data['definitions'][0].get('definition', '').strip()
                        if definition:
                            if progress_callback:
                                progress_callback(f"Getting definition for '{word}'... (from WordsAPI)")
                            return definition, "WordsAPI"
            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
                # No more APIs to try
                pass
            except Exception as e:
                # No more APIs to try
                pass
        
        # All APIs failed for this word
        return "", "None"
    
    def _get_base_word(self, word: str) -> str:
        """Try to find the base/root word by stripping common suffixes."""
        word_lower = word.lower()
        
        # Common suffixes to try removing (in order of likelihood)
        suffixes = [
            # Past tense and past participle
            ('ied', 'y'), ('ied', 'ie'),  # studied -> study, tried -> try
            ('ed', ''),  # pummeled -> pummel, berated -> berate
            # Present participle and gerund - special cases first
            ('ating', 'ate'),  # ingratiating -> ingratiate, creating -> create
            ('ying', 'y'),  # trying -> try
            ('ing', ''),  # searing -> sear, pummeled -> pummel
            # Plural and third person singular
            ('ies', 'y'),  # cities -> city
            ('ies', 'ie'),  # studies -> study
            ('es', ''),  # boxes -> box, rations -> ration
            # Only strip 's' if word is longer than 5 chars (to avoid stripping from words like "tirade", "constellation")
            # This helps with plurals like "rations" -> "ration" but preserves words ending in 's' naturally
            ('s', ''),  # rations -> ration (but not "tirade" -> "tirad")
        ]
        
        # Try removing suffixes
        for suffix, replacement in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                # Special handling for 's' suffix - only strip if word is longer than 5 chars
                # This avoids stripping 's' from words that naturally end in 's' like "tirade", "constellation"
                if suffix == 's' and len(word_lower) <= 5:
                    continue
                
                base = word_lower[:-len(suffix)] + replacement
                if len(base) > 2:  # Ensure base word is meaningful
                    return base
        
        return word_lower
    
    def _fetch_pronunciation_from_api(self, word: str, progress_callback: Optional[callable] = None) -> str:
        """Fetch pronunciation from online dictionary APIs.
        
        Uses the base word (not the original word) for pronunciations.
        
        Args:
            word: Word to fetch pronunciation for
            progress_callback: Optional callback to report progress with source info
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        import json
        
        # Use base word for pronunciation
        base_word = self._get_base_word(word)
        
        # Try base word first with Free Dictionary API
        pronunciation, source = self._try_fetch_pronunciation_free_api(base_word, progress_callback)
        if pronunciation:
            if progress_callback:
                progress_callback(f"Getting pronunciation for '{word}'... (from {source})")
            return pronunciation
        
        # Try other APIs with base word
        pronunciation, source = self._try_fetch_pronunciation_other_apis(base_word, progress_callback)
        if pronunciation:
            if progress_callback:
                progress_callback(f"Getting pronunciation for '{word}'... (from {source})")
            return pronunciation
        
        # If base word failed and it's different from original, try original as fallback
        if base_word != word.lower():
            pronunciation, source = self._try_fetch_pronunciation_free_api(word, progress_callback)
            if pronunciation:
                if progress_callback:
                    progress_callback(f"Getting pronunciation for '{word}'... (from {source})")
                return pronunciation
            pronunciation, source = self._try_fetch_pronunciation_other_apis(word, progress_callback)
            if pronunciation:
                if progress_callback:
                    progress_callback(f"Getting pronunciation for '{word}'... (from {source})")
                return pronunciation
        
        return ""
    
    def _try_fetch_pronunciation_free_api(self, word: str, progress_callback: Optional[callable] = None) -> tuple:
        """Try to fetch pronunciation from Free Dictionary API.
        
        Returns:
            Tuple of (pronunciation, source_name)
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        import json
        
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word.lower())}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'English-Learning-Helper/1.0')
            
            # Reduced timeout from 10 to 5 seconds for faster failure and better performance
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                if data and len(data) > 0:
                    # Get pronunciation (phonetic)
                    phonetic = data[0].get('phonetic', '')
                    if phonetic:
                        return phonetic.strip('/'), "Free Dictionary API"
                    
                    # Try to get from phonetics array
                    phonetics = data[0].get('phonetics', [])
                    if phonetics and len(phonetics) > 0:
                        for phonetic_entry in phonetics:
                            text = phonetic_entry.get('text', '')
                            if text:
                                return text.strip('/'), "Free Dictionary API"
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
            pass
        except Exception as e:
            pass
        
        return "", "None"
    
    def _try_fetch_pronunciation_other_apis(self, word: str, progress_callback: Optional[callable] = None) -> tuple:
        """Try to fetch pronunciation from other APIs (Oxford, WordsAPI).
        
        Returns:
            Tuple of (pronunciation, source_name)
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        import json
        
        # 2. Oxford Dictionary API (if API key is available)
        oxford_app_id = os.getenv('OXFORD_APP_ID')
        oxford_api_key = os.getenv('OXFORD_API_KEY')
        if oxford_app_id and oxford_api_key:
            try:
                url = f"https://od-api.oxforddictionaries.com/api/v2/entries/en-gb/{urllib.parse.quote(word.lower())}"
                req = urllib.request.Request(url)
                req.add_header('app_id', oxford_app_id)
                req.add_header('app_key', oxford_api_key)
                req.add_header('User-Agent', 'English-Learning-Helper/1.0')
                
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    
                    if 'results' in data and len(data['results']) > 0:
                        lexical_entries = data['results'][0].get('lexicalEntries', [])
                        if lexical_entries:
                            entries = lexical_entries[0].get('entries', [])
                            if entries:
                                pronunciations = entries[0].get('pronunciations', [])
                                if pronunciations:
                                    phonetic_spelling = pronunciations[0].get('phoneticSpelling', '')
                                    if phonetic_spelling:
                                        return phonetic_spelling.strip('/'), "Oxford Dictionary API"
            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
                # Try next API
                pass
            except Exception as e:
                # Other errors - try next API
                pass
        
        # 3. WordsAPI (if API key is available)
        words_api_key = os.getenv('WORDS_API_KEY')
        if words_api_key:
            try:
                url = f"https://wordsapiv1.p.rapidapi.com/words/{urllib.parse.quote(word.lower())}"
                req = urllib.request.Request(url)
                req.add_header('X-RapidAPI-Key', words_api_key)
                req.add_header('X-RapidAPI-Host', 'wordsapiv1.p.rapidapi.com')
                req.add_header('User-Agent', 'English-Learning-Helper/1.0')
                
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    
                    if 'pronunciation' in data:
                        pronunciation = data['pronunciation']
                        if isinstance(pronunciation, dict):
                            # Sometimes it's a dict with 'all' key
                            pronunciation = pronunciation.get('all', '')
                        if pronunciation:
                            return str(pronunciation).strip('/'), "WordsAPI"
            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
                # No more APIs to try
                pass
            except Exception as e:
                # No more APIs to try
                pass
        
        # All APIs failed - return empty string
        return "", "None"

    def _generate_mock_sentences(self, word: str, num_sentences: int) -> List[str]:
        """Generate mock sentences for demonstration purposes."""
        mock_templates = [
            f"The concept of {word} has revolutionized our daily lives in many ways.",
            f"I'm really interested in learning more about {word} and its applications.",
            f"Have you heard about the latest developments in {word} research?",
            f"Understanding {word} is crucial for modern innovation and progress.",
            f"Many experts believe that {word} will shape the future of our society.",
            f"The idea of {word} might seem complex at first, but it's actually quite fascinating.",
            f"Companies are investing heavily in {word} to stay competitive in the market.",
            f"Students should learn about {word} to prepare for future careers.",
            f"Scientists are making great advances in the field of {word}.",
            f"The benefits of {word} are becoming more apparent every day.",
        ]

        # Shuffle and select sentences
        import random
        random.shuffle(mock_templates)
        selected = mock_templates[:num_sentences]

        return selected

    def _parse_sentences(self, response_text: str) -> List[str]:
        """Parse sentences from AI response."""
        sentences = []
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() and line[1:3] in ['. ', ') ']):
                # Extract sentence after number
                sentence = line.split('. ', 1)[1] if '. ' in line else line.split(') ', 1)[1] if ') ' in line else line
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)

        return sentences[:3]  # Return up to 3 sentences


class FillInBlankGenerator:
    """Creates fill-in-the-blank tests from sentences."""

    def __init__(self):
        pass

    def create_tests(self, word_sentences: Dict[str, List[str]]) -> List[Dict]:
        """
        Create fill-in-the-blank tests from word-sentence pairs.

        Args:
            word_sentences: Dictionary mapping words to their sentences

        Returns:
            List of test items with blanks and answers
        """
        tests = []

        for word, sentences in word_sentences.items():
            for sentence in sentences:
                # Create a blank by replacing the target word with _____
                # Simple approach: replace the first occurrence of the word
                blank_sentence = re.sub(rf'\b{re.escape(word)}\b', '_____', sentence, flags=re.IGNORECASE)

                # Only create test if the word was actually replaced
                if blank_sentence != sentence:
                    tests.append({
                        'sentence': blank_sentence,
                        'answer': word,
                        'original': sentence
                    })

        return tests

    def split_tests_into_batches(self, tests: List[Dict], batch_size: int = 20) -> List[List[Dict]]:
        """
        Split tests into batches of specified size.

        Args:
            tests: List of test items
            batch_size: Number of tests per batch

        Returns:
            List of batches, each containing up to batch_size tests
        """
        batches = []
        for i in range(0, len(tests), batch_size):
            batches.append(tests[i:i + batch_size])
        return batches

    def run_test(self, tests: List[Dict]) -> None:
        """Run an interactive fill-in-the-blank test."""
        if not tests:
            console.print("[yellow]No tests available![/yellow]")
            return

        score = 0
        total = len(tests)

        console.print(Panel.fit("[bold blue]Fill-in-the-Blank Test[/bold blue]"))
        console.print(f"You will be tested on {total} sentences.\n")

        for i, test in enumerate(tests, 1):
            console.print(f"[bold]{i}.[/bold] {test['sentence']}")

            # Get user input
            user_answer = click.prompt("Your answer").strip().lower()

            if user_answer == test['answer'].lower():
                console.print("[green][CORRECT][/green]")
                score += 1
            else:
                console.print(f"[red][WRONG] The correct answer is: {test['answer']}[/red]")
                console.print(f"[dim]Full sentence: {test['original']}[/dim]")

            console.print()  # Empty line

        # Show final score
        percentage = (score / total) * 100 if total > 0 else 0
        console.print(Panel.fit(f"[bold]Test Complete![/bold]\nScore: {score}/{total} ({percentage:.1f}%)"))


def process_vocabulary_files(
    file_paths: List[str],
    examples_per_word: int = 1,
    output: Optional[str] = None,
    test_batch_size: int = 20,
    words_per_section: int = 20,
    track_new_words: bool = True,
    tracker_file: str = 'processed_words.json',
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Shared function to process vocabulary files. Used by both CLI and GUI.
    
    Args:
        file_paths: List of file paths to process
        examples_per_word: Number of example sentences per word
        output: Output filename (None for auto-generated)
        test_batch_size: Number of questions per test batch
        words_per_section: Number of words per section before test
        track_new_words: Whether to track new words
        tracker_file: File to track processed words
        progress_callback: Optional callback function(progress_message: str) for progress updates
    
    Returns:
        Dictionary with results: {
            'word_data': Dict[str, Dict],
            'all_tests': List[Dict],
            'test_batches': List[List[Dict]],
            'new_words': List[str],
            'duplicate_words': List[str],
            'total_words': int,
            'output': str
        }
    """
    def emit_progress(message: str):
        """Emit progress message via callback or console."""
        if progress_callback:
            progress_callback(message)
        else:
            console.print(f"[dim]{message}[/dim]")
    
    # Helper function to find file with extensions
    def find_file(file_path: str) -> Optional[str]:
        """Try to find file, checking common extensions."""
        if os.path.exists(file_path):
            return file_path
        
        extensions = ['', '.html', '.htm', '.txt', '.md']
        for ext in extensions:
            test_path = file_path + ext
            if os.path.exists(test_path):
                return test_path
        return None
    
    # Process all file paths
    actual_paths = []
    for file_path in file_paths:
        actual_path = find_file(file_path)
        if actual_path:
            actual_paths.append(actual_path)
            emit_progress(f"Found file: {actual_path}")
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    if not actual_paths:
        raise ValueError("No valid files found to process.")
    
    if len(actual_paths) > 1:
        emit_progress(f"Processing {len(actual_paths)} files...")
    
    # Load vocabulary from all files
    loader = VocabularyLoader()
    all_words = []
    
    for actual_path in actual_paths:
        words_from_file = loader.load_file(actual_path)
        all_words.extend(words_from_file)
        emit_progress(f"Loaded {len(words_from_file)} words from {os.path.basename(actual_path)}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)
    
    all_words = unique_words
    
    if not all_words:
        raise ValueError("No vocabulary words found in file(s).")
    
    # Track new words if enabled
    words_to_process = all_words
    duplicate_words = []
    new_words = []
    tracker = None
    
    if track_new_words:
        tracker = WordTracker(tracker_file)
        new_words, duplicate_words = tracker.get_new_words(all_words)
        
        if len(new_words) == 0 and len(duplicate_words) == 0:
            raise ValueError("No new words found! All words have already been processed.")
        
        # Process both new words and duplicates
        words_to_process = new_words + duplicate_words
    
    total_words = len(words_to_process)
    emit_progress(f"Generating explanations for {total_words} words... (from Google AI)")
    
    # Generate explanations and examples for words to process using parallel processing
    generator = SentenceGenerator()
    
    # Debug: Check if Google AI is actually available - ALWAYS show this if in mock mode
    debug_enabled = os.getenv('DEBUG_GOOGLE_AI', '0') == '1'
    if generator.use_mock:
        emit_progress(f"[WARNING] Google AI is in MOCK mode - definitions will come from dictionary APIs")
        emit_progress(f"[WARNING] Check your GOOGLE_API_KEY in .env file")
    elif debug_enabled:
        emit_progress(f"[DEBUG] Google AI initialized: use_mock={generator.use_mock}, has_models={len(generator.models) > 0}")
        if generator.model:
            emit_progress(f"[DEBUG] Using model: {generator.model}")
    
    word_data = {}  # Store explanation and examples for each word
    duplicate_set = set(w.lower() for w in duplicate_words)  # For quick lookup
    
    # Thread-safe progress counter
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list to allow modification in nested function
    
    def process_single_word(word: str) -> tuple:
        """Process a single word - designed for parallel execution."""
        try:
            is_duplicate = word.lower() in duplicate_set
            
            # For duplicates, request different examples (add a note to get varied examples)
            debug_enabled = os.getenv('DEBUG_GOOGLE_AI', '0') == '1'
            
            if is_duplicate and tracker:
                occurrence_count = tracker.get_occurrence_count(word)
                # Generate with a note that this is a repeat word needing different examples
                emit_progress(f"Processing '{word}'... (from Google AI)")
                if debug_enabled:
                    emit_progress(f"[DEBUG] Calling Google AI for '{word}' (duplicate word)")
                data = generator.generate_explanation_and_examples(word, examples_per_word, is_repeat=True)
                data['is_important'] = True
                data['occurrence_count'] = occurrence_count
            else:
                emit_progress(f"Processing '{word}'... (from Google AI)")
                if debug_enabled:
                    emit_progress(f"[DEBUG] Calling Google AI for '{word}'")
                data = generator.generate_explanation_and_examples(word, examples_per_word)
                data['is_important'] = False
            
            # Check what we got from Google AI
            explanation = data.get('explanation', '').strip()
            if debug_enabled:
                emit_progress(f"[DEBUG] Google AI result for '{word}': explanation length={len(explanation)}, has_explanation={bool(explanation)}")
                if explanation:
                    emit_progress(f"[DEBUG] Explanation preview: '{explanation[:100]}...'")
            
            # Ensure definition is included - use original word only (no base word fallback)
            # Only fallback to dictionary API if Google AI completely failed
            if not explanation or 'Unable to fetch definition' in explanation:
                # Google AI failed, try fetching definition with dictionary API as fallback
                if debug_enabled:
                    emit_progress(f"[DEBUG] Google AI failed for '{word}', falling back to dictionary API")
                emit_progress(f"Getting definition for '{word}'... (from dictionary API - Google AI unavailable/failed)")
                # Don't pass emit_progress here to avoid duplicate messages - we already showed the fallback message
                explanation = generator._fetch_definition_from_api(word)
                if explanation and 'Unable to fetch definition' not in explanation:
                    data['explanation'] = explanation
            # Note: If Google AI succeeded, we don't need to show a separate progress message
            # because the "Processing 'word'... (from Google AI)" message already indicates it's using Google AI
            
            # Ensure pronunciation is included - try multiple sources
            pronunciation = data.get('pronunciation', '').strip()
            if not pronunciation:
                # Try VocabularyLoader cache first (already fetched during loading)
                pronunciation = loader.get_pronunciation(word).strip()
                if pronunciation:
                    emit_progress(f"Getting pronunciation for '{word}'... (from cache)")
            if not pronunciation:
                # Last resort: try fetching directly from API one more time
                pronunciation = generator._fetch_pronunciation_from_api(word, emit_progress).strip()
            
            # Store pronunciation in data (always store, even if empty)
            data['pronunciation'] = pronunciation
            
            # Update progress (thread-safe)
            with progress_lock:
                completed_count[0] += 1
                count = completed_count[0]
                if count % 10 == 0 or count == total_words:
                    emit_progress(f"Processed {count}/{total_words} words")
            
            return (word, data, None)
        except Exception as e:
            # Update progress even on error
            with progress_lock:
                completed_count[0] += 1
                count = completed_count[0]
            
            emit_progress(f"WARNING: Failed to generate for '{word}': {e}")
            # Use mock data as fallback
            try:
                is_duplicate = word.lower() in duplicate_set
                data = generator._generate_mock_explanation_and_examples(word, examples_per_word, is_repeat=is_duplicate)
                data['is_important'] = is_duplicate
                if is_duplicate and tracker:
                    data['occurrence_count'] = tracker.get_occurrence_count(word)
                
                # Ensure definition is included - use original word only (no base word fallback)
                explanation = data.get('explanation', '').strip()
                if not explanation or 'Unable to fetch definition' in explanation:
                    # Try fetching definition with original word only
                    explanation = generator._fetch_definition_from_api(word)
                    if explanation and 'Unable to fetch definition' not in explanation:
                        data['explanation'] = explanation
                
                # Ensure pronunciation is included - try multiple sources
                pronunciation = data.get('pronunciation', '').strip()
                if not pronunciation:
                    # Try VocabularyLoader cache first (already fetched during loading)
                    pronunciation = loader.get_pronunciation(word).strip()
                if not pronunciation:
                    # Last resort: try fetching directly from API one more time
                    pronunciation = generator._fetch_pronunciation_from_api(word).strip()
                
                # Store pronunciation in data (always store, even if empty)
                data['pronunciation'] = pronunciation
                
                return (word, data, None)
            except Exception as e2:
                return (word, None, e2)
    
    # Use ThreadPoolExecutor for parallel processing
    # For I/O-bound tasks (API calls), we can use many more threads
    # Increased from 8 to 20 for significantly faster processing
    max_workers = min(20, total_words)  # Don't create more threads than words
    emit_progress(f"Processing with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_word = {executor.submit(process_single_word, word): word for word in words_to_process}
        
        # Collect results as they complete
        for future in as_completed(future_to_word):
            word, data, error = future.result()
            if error:
                emit_progress(f"ERROR: Critical error processing '{word}': {error}")
            elif data:
                word_data[word] = data
    
    emit_progress(f"Generated explanations and examples for all {total_words} words")
    
    # Create fill-in-the-blank tests from examples
    emit_progress("Creating fill-in-the-blank tests...")
    test_generator = FillInBlankGenerator()
    
    # Convert word_data to word_sentences format for test generation
    word_sentences = {word: data['examples'] for word, data in word_data.items()}
    all_tests = test_generator.create_tests(word_sentences)
    
    # Split tests into batches
    test_batches = test_generator.split_tests_into_batches(all_tests, test_batch_size)
    emit_progress(f"Created {len(all_tests)} test questions in {len(test_batches)} batches")
    
    # Determine output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output is None:
        output = "vocabulary_learning_materials.txt"
    
    # Add timestamp to filename before extension
    if os.path.dirname(output):
        # Full path provided
        base_path = os.path.dirname(output)
        filename = os.path.basename(output)
        name, ext = os.path.splitext(filename)
        output = os.path.join(base_path, f"{name}_{timestamp}{ext}")
    else:
        # Just filename provided - use Documents directory as default
        name, ext = os.path.splitext(output)
        output = f"{name}_{timestamp}{ext}"
        # Use Documents directory instead of current working directory
        documents_dir = r"C:\Users\Admin\Documents"
        if os.path.exists(documents_dir):
            output = os.path.join(documents_dir, output)
        else:
            # Fallback to current directory if Documents doesn't exist
            output = os.path.join(os.getcwd(), output)
    
    # Group words into sections - preserve original file order
    # Create word list in the same order as words_to_process (original file order)
    word_list = []
    for word in words_to_process:
        if word in word_data:  # Only include words that were successfully processed
            word_list.append((word, word_data[word]))
    
    word_sections = []
    for i in range(0, len(word_list), words_per_section):
        word_sections.append(word_list[i:i + words_per_section])
    
    # Create a mapping from word to its tests for quick lookup
    word_to_tests = {}
    for test_item in all_tests:
        word = test_item['answer'].lower()
        if word not in word_to_tests:
            word_to_tests[word] = []
        word_to_tests[word].append(test_item)
    
    # Save everything to file
    emit_progress(f"Saving results to: {output}")
    
    with open(output, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("VOCABULARY LEARNING MATERIALS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Vocabulary Words: {total_words}\n")
        if track_new_words and len(duplicate_words) > 0:
            f.write(f"New Words: {len(new_words)}\n")
            f.write(f"Important/Duplicate Words: {len(duplicate_words)} (appeared in previous learning)\n")
        f.write(f"Examples per Word: {examples_per_word}\n")
        f.write(f"Total Test Questions: {len(all_tests)}\n")
        f.write(f"Words per Section: {words_per_section}\n")
        f.write(f"Test Batch Size: {test_batch_size}\n")
        f.write("=" * 80 + "\n\n\n")
        
        # Process each section: words → test → answers
        for section_num, word_section in enumerate(word_sections, 1):
            # Write vocabulary words for this section
            f.write(f"SECTION {section_num}: VOCABULARY WORDS {((section_num - 1) * words_per_section) + 1}-{min(section_num * words_per_section, total_words)}\n")
            f.write("=" * 80 + "\n\n")
            
            section_tests = []
            
            for idx, (word, data) in enumerate(word_section, 1):
                global_idx = (section_num - 1) * words_per_section + idx
                
                # Get pronunciation and format it (strip existing slashes if present)
                pronunciation = data.get('pronunciation', '')
                if pronunciation:
                    # Remove leading/trailing slashes if present
                    pronunciation = pronunciation.strip('/')
                    pronunciation_text = f" /{pronunciation}/"
                else:
                    pronunciation_text = ""
                
                # Mark important/duplicate words
                if data.get('is_important', False):
                    occurrence_count = data.get('occurrence_count', 1)
                    f.write(f"{global_idx}. word: {word.lower()}{pronunciation_text} [IMPORTANT - appeared {occurrence_count} time(s) in previous learning]\n")
                    f.write("-" * 80 + "\n")
                    f.write("NOTE: This word appeared in previous learning sessions. Reviewing with different examples to reinforce understanding.\n\n")
                else:
                    f.write(f"{global_idx}. word: {word.lower()}{pronunciation_text}\n")
                    f.write("-" * 80 + "\n")
                
                f.write(f"explanation:\n{data['explanation']}\n\n")
                f.write("example:\n")
                for example in data['examples']:
                    f.write(f"  • {example}\n")
                f.write("\n")
                
                # Collect tests for words in this section
                if word.lower() in word_to_tests:
                    section_tests.extend(word_to_tests[word.lower()])
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Write test for this section
            if section_tests:
                # Shuffle tests to randomize order (makes it more challenging)
                # Tests are shuffled so they don't appear in the same order as word explanations
                shuffled_tests = section_tests.copy()
                random.shuffle(shuffled_tests)
                
                # Split section tests into batches
                section_test_batches = []
                for i in range(0, len(shuffled_tests), test_batch_size):
                    section_test_batches.append(shuffled_tests[i:i + test_batch_size])
                
                for batch_num, batch in enumerate(section_test_batches, 1):
                    f.write(f"TEST - Section {section_num}, Batch {batch_num} ({len(batch)} questions)\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for i, test_item in enumerate(batch, 1):
                        f.write(f"Question {i}:\n")
                        f.write(f"{test_item['sentence']}\n\n")
                    
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("ANSWERS:\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for i, test_item in enumerate(batch, 1):
                        f.write(f"Question {i}: {test_item['answer']}\n")
                        f.write(f"  Full sentence: {test_item['original']}\n\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n\n")
    
    # Update tracker with processed words
    if track_new_words:
        tracker.add_words(words_to_process)
        tracker.save()
        emit_progress(f"Updated tracker: {tracker.get_stats()['total_processed']} total words tracked")
    
    emit_progress(f"Successfully saved all materials to: {output}")
    
    return {
        'word_data': word_data,
        'all_tests': all_tests,
        'test_batches': test_batches,
        'new_words': new_words,
        'duplicate_words': duplicate_words,
        'total_words': total_words,
        'output': output
    }


@click.group()
def cli():
    """English Learning Helper - Generate sentences and tests from vocabulary words."""
    pass


@cli.command()
@click.argument('file_paths', nargs=-1, type=click.Path(), required=True)
@click.option('--examples-per-word', default=1, help='Number of example sentences per word')
@click.option('--output', default=None, help='Output filename (default: vocabulary_learning_materials.txt)')
@click.option('--test-batch-size', default=20, help='Number of questions per test batch')
@click.option('--words-per-section', default=20, help='Number of words per section before test')
@click.option('--track-new-words/--no-track', default=True, help='Track processed words and only process new ones')
@click.option('--tracker-file', default='processed_words.json', help='File to track processed words')
def process(file_paths: tuple, examples_per_word: int, output: Optional[str], test_batch_size: int, words_per_section: int, track_new_words: bool, tracker_file: str):
    """Process one or more vocabulary files and generate explanations, examples, and tests for ALL words.
    
    You can specify multiple files:
    python english_learner.py process file1.html file2.html file3.txt
    """
    try:
        # Convert tuple to list
        file_paths_list = list(file_paths)
        
        # Use shared processing function
        result = process_vocabulary_files(
            file_paths=file_paths_list,
            examples_per_word=examples_per_word,
            output=output,
            test_batch_size=test_batch_size,
            words_per_section=words_per_section,
            track_new_words=track_new_words,
            tracker_file=tracker_file,
            progress_callback=None  # CLI uses console.print directly
        )
        
        # Display summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  • {result['total_words']} vocabulary words processed")
        if track_new_words:
            if len(result['new_words']) > 0:
                console.print(f"    - {len(result['new_words'])} new words")
            if len(result['duplicate_words']) > 0:
                console.print(f"    - {len(result['duplicate_words'])} important words (duplicates from previous learning)")
        console.print(f"  • {len(result['all_tests'])} test questions created")
        console.print(f"  • {len(result['test_batches'])} test batches ({test_batch_size} questions each)")
        console.print(f"  • Output file: {result['output']}")
        if track_new_words:
            console.print(f"  • Tracker file: {tracker_file}")

    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please set your GOOGLE_API_KEY environment variable.")
    except Exception as e:
        console.print(f"[red][ERROR] An error occurred: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


# Old process function code removed - now using shared process_vocabulary_files() function
    # Helper function to find file with extensions
    def find_file(file_path: str) -> Optional[str]:
        """Try to find file, checking common extensions."""
        if os.path.exists(file_path):
            return file_path
        
        extensions = ['', '.html', '.htm', '.txt', '.md']
        for ext in extensions:
            test_path = file_path + ext
            if os.path.exists(test_path):
                return test_path
        return None
    
    # Process all file paths
    actual_paths = []
    for file_path in file_paths:
        actual_path = find_file(file_path)
        if actual_path:
            actual_paths.append(actual_path)
            console.print(f"[green][OK] Found file: {actual_path}[/green]")
        else:
            console.print(f"[red][ERROR] File not found: {file_path}[/red]")
            console.print("[yellow]Tried extensions: .html, .htm, .txt, .md[/yellow]")
    
    if not actual_paths:
        console.print("[red][ERROR] No valid files found to process.[/red]")
        return
    
    if len(actual_paths) > 1:
        console.print(f"[bold]Processing {len(actual_paths)} files...[/bold]\n")
    
    # Load vocabulary from all files
    loader = VocabularyLoader()
    all_words = []
    file_word_counts = []
    
    for actual_path in actual_paths:
        words_from_file = loader.load_file(actual_path)
        file_word_counts.append((actual_path, len(words_from_file)))
        all_words.extend(words_from_file)
        console.print(f"[dim]Loaded {len(words_from_file)} words from {os.path.basename(actual_path)}[/dim]")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)
    
    all_words = unique_words
    
    console.print(f"\n[bold green]Total unique words from all files: {len(all_words)}[/bold green]")
    if len(actual_paths) > 1:
        console.print(f"[dim]Combined from {len(actual_paths)} files (removed duplicates)[/dim]\n")

    if not all_words:
        console.print("[red][ERROR] No vocabulary words found in file.[/red]")
        return

    # Track new words if enabled
    words_to_process = all_words
    duplicate_words = []
    new_words = []
    tracker = None
    
    if track_new_words:
        tracker = WordTracker(tracker_file)
        stats = tracker.get_stats()
        console.print(f"[dim]Loaded tracker: {stats['total_processed']} words already processed[/dim]")
        
        new_words, duplicate_words = tracker.get_new_words(all_words)
        
        if len(duplicate_words) > 0:
            console.print(f"[bold yellow]Found {len(duplicate_words)} IMPORTANT words (appeared in previous learning)[/bold yellow]")
            for dup_word in duplicate_words[:10]:  # Show first 10
                count = tracker.get_occurrence_count(dup_word)
                word_display = loader.format_word_with_pronunciation(dup_word)
                console.print(f"[dim]  • {word_display} (appeared {count} time(s) before)[/dim]")
            if len(duplicate_words) > 10:
                console.print(f"[dim]  ... and {len(duplicate_words) - 10} more[/dim]")
        
        if len(new_words) == 0 and len(duplicate_words) == 0:
            console.print("[green]No new words found! All words have already been processed.[/green]")
            console.print(f"[dim]Total words in file: {len(all_words)}[/dim]")
            console.print(f"[dim]Processed words: {stats['total_processed']}[/dim]")
            return
        
        # Process both new words and duplicates
        words_to_process = new_words + duplicate_words
        
        if len(new_words) > 0:
            console.print(f"[bold green]Found {len(new_words)} NEW words to process[/bold green]")
        if len(duplicate_words) > 0:
            console.print(f"[bold yellow]Found {len(duplicate_words)} IMPORTANT duplicate words to reinforce[/bold yellow]")
    else:
        console.print(f"[bold green]Found {len(all_words)} vocabulary words[/bold green]")

    total_words = len(words_to_process)
    console.print(f"[bold]Generating explanations and examples for {total_words} words...[/bold]")
    console.print(f"[dim]Using parallel processing with threading for faster execution...[/dim]\n")

    # Generate explanations and examples for words to process using parallel processing
    try:
        generator = SentenceGenerator()
        word_data = {}  # Store explanation and examples for each word
        duplicate_set = set(w.lower() for w in duplicate_words)  # For quick lookup
        
        # Thread-safe progress counter
        progress_lock = threading.Lock()
        completed_count = [0]  # Use list to allow modification in nested function
        
        def process_single_word(word: str) -> tuple:
            """Process a single word - designed for parallel execution."""
            try:
                is_duplicate = word.lower() in duplicate_set
                
                # For duplicates, request different examples (add a note to get varied examples)
                if is_duplicate and tracker:
                    occurrence_count = tracker.get_occurrence_count(word)
                    # Generate with a note that this is a repeat word needing different examples
                    data = generator.generate_explanation_and_examples(word, examples_per_word, is_repeat=True)
                    data['is_important'] = True
                    data['occurrence_count'] = occurrence_count
                else:
                    data = generator.generate_explanation_and_examples(word, examples_per_word)
                    data['is_important'] = False
                
                # Ensure definition is included - use original word only (no base word fallback)
                explanation = data.get('explanation', '').strip()
                if not explanation or 'Unable to fetch definition' in explanation:
                    # Try fetching definition with original word only
                    explanation = generator._fetch_definition_from_api(word)
                    if explanation and 'Unable to fetch definition' not in explanation:
                        data['explanation'] = explanation
                
                # Ensure pronunciation is included - try multiple sources
                pronunciation = data.get('pronunciation', '').strip()
                if not pronunciation:
                    # Try VocabularyLoader cache first (already fetched during loading)
                    pronunciation = loader.get_pronunciation(word).strip()
                if not pronunciation:
                    # Last resort: try fetching directly from API one more time
                    pronunciation = generator._fetch_pronunciation_from_api(word).strip()
                
                # Store pronunciation in data (always store, even if empty)
                data['pronunciation'] = pronunciation
                
                # Update progress (thread-safe)
                with progress_lock:
                    completed_count[0] += 1
                    count = completed_count[0]
                    if count % 10 == 0 or count == total_words:
                        console.print(f"[green][OK] Processed {count}/{total_words} words[/green]")
                
                return (word, data, None)
            except Exception as e:
                # Update progress even on error
                with progress_lock:
                    completed_count[0] += 1
                    count = completed_count[0]
                
                console.print(f"[yellow][WARN] Failed to generate for '{word}': {e}[/yellow]")
                # Use mock data as fallback
                try:
                    is_duplicate = word.lower() in duplicate_set
                    data = generator._generate_mock_explanation_and_examples(word, examples_per_word, is_repeat=is_duplicate)
                    data['is_important'] = is_duplicate
                    if is_duplicate and tracker:
                        data['occurrence_count'] = tracker.get_occurrence_count(word)
                    
                    # Ensure definition is included - use original word only (no base word fallback)
                    explanation = data.get('explanation', '').strip()
                    if not explanation or 'Unable to fetch definition' in explanation:
                        # Try fetching definition with original word only
                        explanation = generator._fetch_definition_from_api(word)
                        if explanation and 'Unable to fetch definition' not in explanation:
                            data['explanation'] = explanation
                    
                    # Ensure pronunciation is included - try multiple sources
                    pronunciation = data.get('pronunciation', '').strip()
                    if not pronunciation:
                        # Try VocabularyLoader cache first (already fetched during loading)
                        pronunciation = loader.get_pronunciation(word).strip()
                    if not pronunciation:
                        # Last resort: try fetching directly from API one more time
                        pronunciation = generator._fetch_pronunciation_from_api(word).strip()
                    
                    # Store pronunciation in data (always store, even if empty)
                    data['pronunciation'] = pronunciation
                    
                    return (word, data, None)
                except Exception as e2:
                    return (word, None, e2)
        
        # Use ThreadPoolExecutor for parallel processing
        # For I/O-bound tasks (API calls), we can use many more threads
        # Increased from 8 to 20 for significantly faster processing
        max_workers = min(20, total_words)  # Don't create more threads than words
        console.print(f"[dim]Processing with {max_workers} parallel workers for maximum speed...[/dim]\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_word = {executor.submit(process_single_word, word): word for word in words_to_process}
            
            # Collect results as they complete
            for future in as_completed(future_to_word):
                word, data, error = future.result()
                if error:
                    console.print(f"[red][ERROR] Critical error processing '{word}': {error}[/red]")
                elif data:
                    word_data[word] = data

        console.print(f"\n[bold green][OK] Generated explanations and examples for all {total_words} words[/bold green]")

        # Create fill-in-the-blank tests from examples
        console.print(f"\n[bold]Creating fill-in-the-blank tests...[/bold]")
        test_generator = FillInBlankGenerator()
        
        # Convert word_data to word_sentences format for test generation
        word_sentences = {word: data['examples'] for word, data in word_data.items()}
        all_tests = test_generator.create_tests(word_sentences)
        
        # Split tests into batches
        test_batches = test_generator.split_tests_into_batches(all_tests, test_batch_size)
        console.print(f"[green][OK] Created {len(all_tests)} test questions in {len(test_batches)} batches[/green]")

        # Determine output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output is None:
            output = "vocabulary_learning_materials.txt"
        
        # Add timestamp to filename before extension
        if os.path.dirname(output):
            # Full path provided
            base_path = os.path.dirname(output)
            filename = os.path.basename(output)
            name, ext = os.path.splitext(filename)
            output = os.path.join(base_path, f"{name}_{timestamp}{ext}")
        else:
            # Just filename provided - use Documents directory as default
            name, ext = os.path.splitext(output)
            output = f"{name}_{timestamp}{ext}"
            # Use Documents directory instead of current working directory
            documents_dir = r"C:\Users\Admin\Documents"
            if os.path.exists(documents_dir):
                output = os.path.join(documents_dir, output)
            else:
                # Fallback to current directory if Documents doesn't exist
                output = os.path.join(os.getcwd(), output)

        # Group words into sections - preserve original file order
        # Create word list in the same order as words_to_process (original file order)
        word_list = []
        for word in words_to_process:
            if word in word_data:  # Only include words that were successfully processed
                word_list.append((word, word_data[word]))
        
        word_sections = []
        for i in range(0, len(word_list), words_per_section):
            word_sections.append(word_list[i:i + words_per_section])
        
        # Create a mapping from word to its tests for quick lookup
        word_to_tests = {}
        for test_item in all_tests:
            word = test_item['answer'].lower()
            if word not in word_to_tests:
                word_to_tests[word] = []
            word_to_tests[word].append(test_item)
        
        # Save everything to file
        console.print(f"\n[bold]Saving results to: {output}[/bold]")
        
        with open(output, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("VOCABULARY LEARNING MATERIALS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Vocabulary Words: {total_words}\n")
            if track_new_words and len(duplicate_words) > 0:
                f.write(f"New Words: {len(new_words)}\n")
                f.write(f"Important/Duplicate Words: {len(duplicate_words)} (appeared in previous learning)\n")
            f.write(f"Examples per Word: {examples_per_word}\n")
            f.write(f"Total Test Questions: {len(all_tests)}\n")
            f.write(f"Words per Section: {words_per_section}\n")
            f.write(f"Test Batch Size: {test_batch_size}\n")
            f.write("=" * 80 + "\n\n\n")
            
            # Process each section: words → test → answers
            for section_num, word_section in enumerate(word_sections, 1):
                # Write vocabulary words for this section
                f.write(f"SECTION {section_num}: VOCABULARY WORDS {((section_num - 1) * words_per_section) + 1}-{min(section_num * words_per_section, total_words)}\n")
                f.write("=" * 80 + "\n\n")
                
                section_tests = []
                
                for idx, (word, data) in enumerate(word_section, 1):
                    global_idx = (section_num - 1) * words_per_section + idx
                    
                    # Get pronunciation and format it (strip existing slashes if present)
                    pronunciation = data.get('pronunciation', '')
                    if pronunciation:
                        # Remove leading/trailing slashes if present
                        pronunciation = pronunciation.strip('/')
                        pronunciation_text = f" /{pronunciation}/"
                    else:
                        pronunciation_text = ""
                    
                    # Mark important/duplicate words
                    if data.get('is_important', False):
                        occurrence_count = data.get('occurrence_count', 1)
                        f.write(f"{global_idx}. word: {word.lower()}{pronunciation_text} [IMPORTANT - appeared {occurrence_count} time(s) in previous learning]\n")
                        f.write("-" * 80 + "\n")
                        f.write("NOTE: This word appeared in previous learning sessions. Reviewing with different examples to reinforce understanding.\n\n")
                    else:
                        f.write(f"{global_idx}. word: {word.lower()}{pronunciation_text}\n")
                        f.write("-" * 80 + "\n")
                    
                    f.write(f"explanation:\n{data['explanation']}\n\n")
                    f.write("example:\n")
                    for example in data['examples']:
                        f.write(f"  • {example}\n")
                    f.write("\n")
                    
                    # Collect tests for words in this section
                    if word.lower() in word_to_tests:
                        section_tests.extend(word_to_tests[word.lower()])
                
                f.write("\n" + "=" * 80 + "\n\n")
                
                # Write test for this section
                if section_tests:
                    # Shuffle tests to randomize order (makes it more challenging)
                    # Tests are shuffled so they don't appear in the same order as word explanations
                    shuffled_tests = section_tests.copy()
                    random.shuffle(shuffled_tests)
                    
                    # Split section tests into batches
                    section_test_batches = []
                    for i in range(0, len(shuffled_tests), test_batch_size):
                        section_test_batches.append(shuffled_tests[i:i + test_batch_size])
                    
                    for batch_num, batch in enumerate(section_test_batches, 1):
                        f.write(f"TEST - Section {section_num}, Batch {batch_num} ({len(batch)} questions)\n")
                        f.write("-" * 80 + "\n\n")
                        
                        for i, test_item in enumerate(batch, 1):
                            f.write(f"Question {i}:\n")
                            f.write(f"{test_item['sentence']}\n\n")
                        
                        f.write("\n" + "-" * 80 + "\n")
                        f.write("ANSWERS:\n")
                        f.write("-" * 80 + "\n\n")
                        
                        for i, test_item in enumerate(batch, 1):
                            f.write(f"Question {i}: {test_item['answer']}\n")
                            f.write(f"  Full sentence: {test_item['original']}\n\n")
                        
                        f.write("\n" + "=" * 80 + "\n\n\n")
        
        # Update tracker with processed words
        if track_new_words:
            tracker.add_words(words_to_process)
            tracker.save()
            stats = tracker.get_stats()
            console.print(f"[green][OK] Updated tracker: {stats['total_processed']} total words tracked[/green]")
        
        console.print(f"[bold green][OK] Successfully saved all materials to: {output}[/bold green]")
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  • {total_words} vocabulary words processed")
        if track_new_words:
            if len(new_words) > 0:
                console.print(f"    - {len(new_words)} new words")
            if len(duplicate_words) > 0:
                console.print(f"    - {len(duplicate_words)} important words (duplicates from previous learning)")
            skipped = len(all_words) - len(words_to_process)
            if skipped > 0:
                console.print(f"  • {skipped} words skipped (already processed)")
        console.print(f"  • {len(all_tests)} test questions created")
        console.print(f"  • {len(test_batches)} test batches ({test_batch_size} questions each)")
        console.print(f"  • Output file: {output}")
        if track_new_words:
            console.print(f"  • Tracker file: {tracker_file}")

    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please set your GOOGLE_API_KEY environment variable.")
    except Exception as e:
        console.print(f"[red][ERROR] An error occurred: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@cli.command()
@click.argument('word')
@click.option('--count', default=3, help='Number of sentences to generate')
def word(word: str, count: int):
    """Generate sentences for a specific word."""
    try:
        generator = SentenceGenerator()

        if generator.use_mock:
            sentences = generator._generate_mock_sentences(word, count)
            console.print(f"[bold blue]Mock sentences for '{word}':[/bold blue]")
            for i, sentence in enumerate(sentences, 1):
                console.print(f"{i}. {sentence}")
        else:
            prompt = f"""
            Generate {count} natural English sentences that native speakers would use,
            incorporating the vocabulary word "{word}" in context.

            Requirements:
            - Each sentence should demonstrate proper usage of "{word}"
            - Sentences should be natural and conversational
            - Vary the sentence structures

            Format your response as a numbered list.
            """

            text = generator._try_generate_content(prompt)

            console.print(f"[bold blue]Sentences for '{word}':[/bold blue]")
            console.print(text)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def check_new_words_and_process(
    file_paths: List[str],
    tracker_file: str = 'processed_words.json',
    output: Optional[str] = None,
    examples_per_word: int = 1,
    test_batch_size: int = 20,
    words_per_section: int = 20,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Shared function to check for new words and generate learning materials for them.
    Used by both CLI and GUI.
    
    Args:
        file_paths: List of file paths to check
        tracker_file: File to track processed words
        output: Output filename (None for auto-generated with checknew prefix)
        examples_per_word: Number of example sentences per word
        test_batch_size: Number of questions per test batch
        words_per_section: Words per section before test
        progress_callback: Optional callback function(progress_message: str) for progress updates
    
    Returns:
        Dictionary with results: {
            'word_data': Dict[str, Dict],
            'all_tests': List[Dict],
            'test_batches': List[List[Dict]],
            'new_words': List[str],
            'duplicate_words': List[str],
            'total_words': int,
            'output': str
        }
    """
    def emit_progress(message: str):
        """Emit progress message via callback or console."""
        if progress_callback:
            progress_callback(message)
        else:
            console.print(f"[dim]{message}[/dim]")
    
    # Helper function to find file with extensions
    def find_file(file_path: str) -> Optional[str]:
        """Try to find file, checking common extensions."""
        if os.path.exists(file_path):
            return file_path
        
        extensions = ['', '.html', '.htm', '.txt', '.md']
        for ext in extensions:
            test_path = file_path + ext
            if os.path.exists(test_path):
                return test_path
        return None
    
    # Process all file paths
    actual_paths = []
    for file_path in file_paths:
        actual_path = find_file(file_path)
        if actual_path:
            actual_paths.append(actual_path)
            emit_progress(f"Found file: {actual_path}")
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    if not actual_paths:
        raise ValueError("No valid files found to process.")
    
    # Load vocabulary WITHOUT fetching pronunciations (optimization for check-new)
    loader = VocabularyLoader()
    all_words = []
    
    for actual_path in actual_paths:
        words_from_file = loader.load_file(actual_path, fetch_pronunciations=False)
        all_words.extend(words_from_file)
        emit_progress(f"Loaded {len(words_from_file)} words from {os.path.basename(actual_path)}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)
    
    all_words = unique_words
    
    if not all_words:
        raise ValueError("No vocabulary words found in file(s).")
    
    # Check for new words FIRST (before fetching pronunciations)
    tracker = WordTracker(tracker_file)
    stats = tracker.get_stats()
    
    new_words, duplicate_words = tracker.get_new_words(all_words)
    
    emit_progress(f"Word Analysis: Total={len(all_words)}, Already processed={len(duplicate_words)}, New={len(new_words)}")
    
    # Only fetch pronunciations for NEW words (optimization)
    if new_words:
        emit_progress(f"Fetching pronunciations for {len(new_words)} new words only... (from dictionary API)")
        loader._fetch_pronunciations_batch(new_words)
    
    if not new_words:
        emit_progress("No new words found! All words have already been processed.")
        return {
            'word_data': {},
            'all_tests': [],
            'test_batches': [],
            'new_words': [],
            'duplicate_words': duplicate_words,
            'total_words': 0,
            'output': None
        }
    
    # Process new words to generate learning materials
    emit_progress(f"Found {len(new_words)} NEW words - generating learning materials...")
    emit_progress(f"Generating explanations and examples for {len(new_words)} words...")
    
    generator = SentenceGenerator()
    word_data = {}  # Store explanation and examples for each word
    
    # Thread-safe progress counter
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list to allow modification in nested function
    
    def process_single_word(word: str) -> tuple:
        """Process a single word - designed for parallel execution."""
        try:
            data = generator.generate_explanation_and_examples(word, examples_per_word)
            data['is_important'] = False
            
            # Ensure definition is included - use original word only (no base word fallback)
            explanation = data.get('explanation', '').strip()
            if not explanation or 'Unable to fetch definition' in explanation:
                # Try fetching definition with original word only
                # Pass emit_progress as callback to show source
                explanation = generator._fetch_definition_from_api(word, emit_progress)
                if explanation and 'Unable to fetch definition' not in explanation:
                    data['explanation'] = explanation
            
            # Ensure pronunciation is included - try multiple sources
            pronunciation = data.get('pronunciation', '').strip()
            if not pronunciation:
                # Try VocabularyLoader cache first (already fetched during loading)
                pronunciation = loader.get_pronunciation(word).strip()
                if pronunciation:
                    emit_progress(f"Getting pronunciation for '{word}'... (from cache)")
            if not pronunciation:
                # Last resort: try fetching directly from API one more time
                pronunciation = generator._fetch_pronunciation_from_api(word, emit_progress).strip()
            
            # Store pronunciation in data (always store, even if empty)
            data['pronunciation'] = pronunciation
            
            # Update progress (thread-safe)
            with progress_lock:
                completed_count[0] += 1
                count = completed_count[0]
                if count % 10 == 0 or count == len(new_words):
                    emit_progress(f"Processed {count}/{len(new_words)} words")
            
            return (word, data, None)
        except Exception as e:
            # Update progress even on error
            with progress_lock:
                completed_count[0] += 1
                count = completed_count[0]
            
            emit_progress(f"WARNING: Failed to generate for '{word}': {e}")
            # Use mock data as fallback
            try:
                data = generator._generate_mock_explanation_and_examples(word, examples_per_word, is_repeat=False)
                data['is_important'] = False
                
                # Ensure definition is included - use original word only (no base word fallback)
                explanation = data.get('explanation', '').strip()
                if not explanation or 'Unable to fetch definition' in explanation:
                    # Try fetching definition with original word only
                    explanation = generator._fetch_definition_from_api(word)
                    if explanation and 'Unable to fetch definition' not in explanation:
                        data['explanation'] = explanation
                
                # Ensure pronunciation is included - try multiple sources
                pronunciation = data.get('pronunciation', '').strip()
                if not pronunciation:
                    # Try VocabularyLoader cache first (already fetched during loading)
                    pronunciation = loader.get_pronunciation(word).strip()
                if not pronunciation:
                    # Last resort: try fetching directly from API one more time
                    pronunciation = generator._fetch_pronunciation_from_api(word).strip()
                
                # Store pronunciation in data (always store, even if empty)
                data['pronunciation'] = pronunciation
                
                return (word, data, None)
            except Exception as e2:
                return (word, None, e2)
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(20, len(new_words))  # Don't create more threads than words
    emit_progress(f"Processing with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_word = {executor.submit(process_single_word, word): word for word in new_words}
        
        # Collect results as they complete
        for future in as_completed(future_to_word):
            word, data, error = future.result()
            if error:
                emit_progress(f"ERROR: Critical error processing '{word}': {error}")
            elif data:
                word_data[word] = data
    
    emit_progress(f"Generated explanations and examples for all {len(new_words)} words")
    
    # Create fill-in-the-blank tests from examples
    emit_progress("Creating fill-in-the-blank tests...")
    test_generator = FillInBlankGenerator()
    
    # Convert word_data to word_sentences format for test generation
    word_sentences = {word: data['examples'] for word, data in word_data.items()}
    all_tests = test_generator.create_tests(word_sentences)
    
    # Split tests into batches
    test_batches = test_generator.split_tests_into_batches(all_tests, test_batch_size)
    emit_progress(f"Created {len(all_tests)} test questions in {len(test_batches)} batches")
    
    # Determine output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output is None:
        # Default filename: vocabulary_learning_materials_checknew_YYYYMMDD_HHMMSS.txt
        # Use Documents directory as default
        documents_dir = r"C:\Users\Admin\Documents"
        if os.path.exists(documents_dir):
            output = os.path.join(documents_dir, f"vocabulary_learning_materials_checknew_{timestamp}.txt")
        else:
            # Fallback to current directory if Documents doesn't exist
            output = f"vocabulary_learning_materials_checknew_{timestamp}.txt"
    else:
        # User specified output - check if it already has a timestamp
        # If the filename already contains a timestamp pattern (YYYYMMDD_HHMMSS), don't add another
        import re
        timestamp_pattern = r'\d{8}_\d{6}'
        
        if os.path.dirname(output):
            # Full path provided
            base_path = os.path.dirname(output)
            filename = os.path.basename(output)
            name, ext = os.path.splitext(filename)
            
            # Check if timestamp already exists in filename
            if re.search(timestamp_pattern, name):
                # Timestamp already present, use as-is (but ensure checknew indicator)
                if 'checknew' not in name.lower() and 'incremental' not in name.lower():
                    # Add checknew but don't add another timestamp
                    output = os.path.join(base_path, f"{name}_checknew{ext}")
                else:
                    # Already has checknew and timestamp, use as-is
                    output = os.path.join(base_path, filename)
            else:
                # No timestamp, add checknew and timestamp
                if 'checknew' not in name.lower() and 'incremental' not in name.lower():
                    output = os.path.join(base_path, f"{name}_checknew_{timestamp}{ext}")
                else:
                    output = os.path.join(base_path, f"{name}_{timestamp}{ext}")
        else:
            # Just filename provided
            name, ext = os.path.splitext(output)
            
            # Check if timestamp already exists in filename
            if re.search(timestamp_pattern, name):
                # Timestamp already present, use as-is (but ensure checknew indicator)
                if 'checknew' not in name.lower() and 'incremental' not in name.lower():
                    # Add checknew but don't add another timestamp
                    output = f"{name}_checknew{ext}"
                else:
                    # Already has checknew and timestamp, use as-is
                    output = output
            else:
                # No timestamp, add checknew and timestamp
                if 'checknew' not in name.lower() and 'incremental' not in name.lower():
                    output = f"{name}_checknew_{timestamp}{ext}"
                else:
                    output = f"{name}_{timestamp}{ext}"
    
    # Ensure output is in Documents directory if no path specified
    if not os.path.dirname(output):
        documents_dir = r"C:\Users\Admin\Documents"
        if os.path.exists(documents_dir):
            output = os.path.join(documents_dir, output)
        else:
            # Fallback to current directory if Documents doesn't exist
            output = os.path.join(os.getcwd(), output)
    
    # Group words into sections - preserve original file order
    word_list = []
    for word in new_words:
        if word in word_data:  # Only include words that were successfully processed
            word_list.append((word, word_data[word]))
    
    word_sections = []
    for i in range(0, len(word_list), words_per_section):
        word_sections.append(word_list[i:i + words_per_section])
    
    # Create a mapping from word to its tests for quick lookup
    word_to_tests = {}
    for test_item in all_tests:
        word = test_item['answer'].lower()
        if word not in word_to_tests:
            word_to_tests[word] = []
        word_to_tests[word].append(test_item)
    
    # Save everything to file
    emit_progress(f"Saving results to: {output}")
    
    with open(output, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("VOCABULARY LEARNING MATERIALS - INCREMENTAL/CHECK-NEW\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total New Vocabulary Words: {len(new_words)}\n")
        f.write(f"Examples per Word: {examples_per_word}\n")
        f.write(f"Total Test Questions: {len(all_tests)}\n")
        f.write(f"Words per Section: {words_per_section}\n")
        f.write(f"Test Batch Size: {test_batch_size}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n\n")
        
        # Process each section: words → test → answers
        for section_num, word_section in enumerate(word_sections, 1):
            # Write vocabulary words for this section
            f.write(f"SECTION {section_num}: VOCABULARY WORDS {((section_num - 1) * words_per_section) + 1}-{min(section_num * words_per_section, len(new_words))}\n")
            f.write("=" * 80 + "\n\n")
            
            section_tests = []
            
            for idx, (word, data) in enumerate(word_section, 1):
                global_idx = (section_num - 1) * words_per_section + idx
                
                # Get pronunciation and format it (strip existing slashes if present)
                pronunciation = data.get('pronunciation', '')
                if pronunciation:
                    # Remove leading/trailing slashes if present
                    pronunciation = pronunciation.strip('/')
                    pronunciation_text = f" /{pronunciation}/"
                else:
                    pronunciation_text = ""
                
                f.write(f"{global_idx}. word: {word.lower()}{pronunciation_text}\n")
                f.write("-" * 80 + "\n")
                
                f.write(f"explanation:\n{data['explanation']}\n\n")
                f.write("example:\n")
                for example in data['examples']:
                    f.write(f"  • {example}\n")
                f.write("\n")
                
                # Collect tests for words in this section
                if word.lower() in word_to_tests:
                    section_tests.extend(word_to_tests[word.lower()])
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Write test for this section
            if section_tests:
                # Shuffle tests to randomize order (makes it more challenging)
                shuffled_tests = section_tests.copy()
                random.shuffle(shuffled_tests)
                
                # Split section tests into batches
                section_test_batches = []
                for i in range(0, len(shuffled_tests), test_batch_size):
                    section_test_batches.append(shuffled_tests[i:i + test_batch_size])
                
                for batch_num, batch in enumerate(section_test_batches, 1):
                    f.write(f"TEST - Section {section_num}, Batch {batch_num} ({len(batch)} questions)\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for i, test_item in enumerate(batch, 1):
                        f.write(f"Question {i}:\n")
                        f.write(f"{test_item['sentence']}\n\n")
                    
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("ANSWERS:\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for i, test_item in enumerate(batch, 1):
                        f.write(f"Question {i}: {test_item['answer']}\n")
                        f.write(f"  Full sentence: {test_item['original']}\n\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n\n")
    
    # Update tracker with processed words
    emit_progress("Updating word tracker...")
    tracker.add_words(new_words)
    tracker.save()
    stats = tracker.get_stats()
    emit_progress(f"Added {len(new_words)} words to tracker (total: {stats['total_processed']})")
    emit_progress(f"Successfully saved all materials to: {output}")
    
    return {
        'word_data': word_data,
        'all_tests': all_tests,
        'test_batches': test_batches,
        'new_words': new_words,
        'duplicate_words': duplicate_words,
        'total_words': len(new_words),
        'output': output
    }


@cli.command()
@click.argument('file_path', type=click.Path())
@click.option('--tracker-file', default='processed_words.json', help='File to track processed words')
@click.option('--output', default=None, help='Output filename (default: auto-generated with checknew/incremental prefix and timestamp)')
@click.option('--examples-per-word', default=1, help='Number of example sentences per word')
@click.option('--test-batch-size', default=20, help='Number of questions per test batch')
@click.option('--words-per-section', default=20, help='Words per section before test')
def check_new(file_path: str, tracker_file: str, output: Optional[str], examples_per_word: int, test_batch_size: int, words_per_section: int):
    """Check for new words in a vocabulary file and generate learning materials for them."""
    try:
        # Use shared function
        result = check_new_words_and_process(
            file_paths=[file_path],
            tracker_file=tracker_file,
            output=output,
            examples_per_word=examples_per_word,
            test_batch_size=test_batch_size,
            words_per_section=words_per_section,
            progress_callback=None  # CLI uses console.print directly
        )
        
        if result['total_words'] == 0:
            # No new words - already handled in shared function
            return
        
        # Display summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  • {result['total_words']} new vocabulary words processed")
        console.print(f"  • {len(result['all_tests'])} test questions created")
        console.print(f"  • {len(result['test_batches'])} test batches ({test_batch_size} questions each)")
        console.print(f"  • Output file: {result['output']}")
        console.print(f"  • Tracker file: {tracker_file} (updated with {result['total_words']} new words)")
        
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please set your GOOGLE_API_KEY environment variable.")
    except Exception as e:
        console.print(f"[red][ERROR] An error occurred: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


def clear_word_tracker(tracker_file: str = 'processed_words.json', skip_confirmation: bool = False) -> Dict:
    """
    Shared function to clear/reset the word tracking cache file.
    Used by both CLI and GUI.
    
    Args:
        tracker_file: Path to the tracker file
        skip_confirmation: If True, skip confirmation (for GUI which handles its own confirmation)
    
    Returns:
        Dictionary with results: {
            'success': bool,
            'words_removed': int,
            'tracker_file': str,
            'message': str
        }
    """
    if not os.path.exists(tracker_file):
        return {
            'success': False,
            'words_removed': 0,
            'tracker_file': tracker_file,
            'message': f"Tracker file '{tracker_file}' does not exist. Nothing to clear."
        }
    
    tracker = WordTracker(tracker_file)
    stats_before = tracker.get_stats()
    words_count = stats_before['total_processed']
    
    tracker.clear()
    
    return {
        'success': True,
        'words_removed': words_count,
        'tracker_file': tracker_file,
        'message': f"Tracker cleared successfully! Removed {words_count} tracked words."
    }


@cli.command()
@click.option('--tracker-file', default='processed_words.json', help='File to track processed words')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def clear_tracker(tracker_file: str, confirm: bool):
    """Clear/reset the word tracking cache file."""
    if not os.path.exists(tracker_file):
        console.print(f"[yellow]Tracker file '{tracker_file}' does not exist. Nothing to clear.[/yellow]")
        return
    
    if not confirm:
        console.print(f"[yellow]WARNING: This will delete all tracked words in '{tracker_file}'[/yellow]")
        response = click.prompt("Are you sure you want to clear the tracker? (yes/no)", default="no")
        if response.lower() not in ['yes', 'y']:
            console.print("[yellow]Cancelled. Tracker file not cleared.[/yellow]")
            return
    
    result = clear_word_tracker(tracker_file, skip_confirmation=True)
    
    if result['success']:
        console.print(f"[bold green][OK] {result['message']}[/bold green]")
        console.print(f"[dim]Removed {result['words_removed']} tracked words[/dim]")
        console.print(f"[dim]File deleted: {result['tracker_file']}[/dim]")
    else:
        console.print(f"[yellow]{result['message']}[/yellow]")


if __name__ == '__main__':
    cli()
