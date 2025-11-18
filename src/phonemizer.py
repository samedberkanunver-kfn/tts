"""
Turkish Phonemizer Module for StyleTTS2 Fine-Tuning

This module provides Turkish text-to-phoneme conversion using espeak-ng backend.
Handles Turkish-specific character normalization, text preprocessing, and phoneme
vocabulary management.

Requirements:
    - espeak-ng system dependency (install: brew install espeak-ng)
    - phonemizer Python package

Author: Claude Code
"""

import re
import json
from typing import List, Dict, Optional, Set
from pathlib import Path
from collections import Counter

try:
    from phonemizer.backend import EspeakBackend
    from phonemizer import phonemize
except ImportError:
    raise ImportError(
        "phonemizer package not found. Install: pip install phonemizer\n"
        "Also install espeak-ng: brew install espeak-ng"
    )


class TurkishPhonemizer:
    """
    Turkish text-to-phoneme converter using espeak-ng backend.

    Handles Turkish-specific preprocessing including:
    - Turkish lowercase conversion (I→ı, İ→i)
    - Number expansion
    - Abbreviation normalization
    - Special character handling
    - Phoneme vocabulary building

    Attributes:
        backend (EspeakBackend): espeak-ng phonemization backend
        phoneme_vocab (Dict[str, int]): Phoneme to index mapping
        index_to_phoneme (Dict[int, str]): Index to phoneme mapping
    """

    # Turkish alphabet
    TURKISH_LOWER = "abcçdefgğhıijklmnoöprsştuüvyz"
    TURKISH_UPPER = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"

    # Common Turkish abbreviations
    ABBREVIATIONS = {
        "Dr.": "Doktor",
        "Prof.": "Profesör",
        "Yrd.": "Yardımcı",
        "Doç.": "Doçent",
        "Sn.": "Sayın",
        "vb.": "ve benzeri",
        "vs.": "ve saire",
        "kr.": "kuruş",
        "TL": "Türk Lirası",
    }

    # Number words in Turkish
    ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
    TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
    HUNDREDS = ["", "yüz", "ikiüz", "üçüz", "dörtyüz", "beşyüz", "altıyüz", "yediyüz", "sekizyüz", "dokuzyüz"]
    SCALES = ["", "bin", "milyon", "milyar", "trilyon"]

    def __init__(
        self,
        language: str = "tr",
        preserve_punctuation: bool = True,
        with_stress: bool = True,
        phoneme_vocab: Optional[Dict[str, int]] = None
    ):
        """
        Initialize Turkish phonemizer.

        Args:
            language: Language code for espeak-ng (default: "tr" for Turkish)
            preserve_punctuation: Whether to keep punctuation marks
            with_stress: Whether to include stress markers
            phoneme_vocab: Pre-built phoneme vocabulary (optional)
        """
        try:
            self.backend = EspeakBackend(
                language=language,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize espeak-ng backend: {e}\n"
                "Make sure espeak-ng is installed: brew install espeak-ng"
            )

        # Phoneme vocabulary
        self.phoneme_vocab: Dict[str, int] = phoneme_vocab or {}
        self.index_to_phoneme: Dict[int, str] = {v: k for k, v in self.phoneme_vocab.items()}

        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"

    def normalize_text(self, text: str) -> str:
        """
        Normalize Turkish text for phonemization.

        Handles:
        - Turkish lowercase conversion (I→ı, İ→i)
        - Number expansion (123 → yüz yirmi üç)
        - Abbreviation expansion (Dr. → Doktor)
        - Special character removal

        Args:
            text: Raw Turkish text

        Returns:
            Normalized text ready for phonemization
        """
        # Turkish-aware lowercase conversion
        text = text.replace('I', 'ı').replace('İ', 'i')
        text = text.lower()

        # Expand abbreviations
        for abbr, full in self.ABBREVIATIONS.items():
            text = re.sub(
                r'\b' + re.escape(abbr) + r'\b',
                full.lower(),
                text,
                flags=re.IGNORECASE
            )

        # Expand numbers
        text = re.sub(r'\b\d+\b', self._number_to_words, text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _number_to_words(self, match: re.Match) -> str:
        """
        Convert numeric string to Turkish words.

        Examples:
            123 → "yüz yirmi üç"
            1000 → "bin"
            5432 → "beş bin dört yüz otuz iki"

        Args:
            match: Regex match object containing the number

        Returns:
            Number as Turkish words
        """
        num = int(match.group())

        if num == 0:
            return "sıfır"

        if num < 0:
            return "eksi " + self._convert_number(abs(num))

        return self._convert_number(num)

    def _convert_number(self, num: int) -> str:
        """
        Convert positive integer to Turkish words.

        Args:
            num: Positive integer

        Returns:
            Number as Turkish words
        """
        if num == 0:
            return ""

        # Handle 1-999
        if num < 1000:
            result = []

            # Hundreds
            hundreds = num // 100
            if hundreds > 0:
                if hundreds == 1:
                    result.append("yüz")
                else:
                    result.append(self.ONES[hundreds] + "yüz")

            # Tens
            tens = (num % 100) // 10
            if tens > 0:
                result.append(self.TENS[tens])

            # Ones
            ones = num % 10
            if ones > 0:
                result.append(self.ONES[ones])

            return " ".join(result)

        # Handle 1000+
        if num < 1000000:
            thousands = num // 1000
            remainder = num % 1000

            if thousands == 1:
                result = "bin"
            else:
                result = self._convert_number(thousands) + " bin"

            if remainder > 0:
                result += " " + self._convert_number(remainder)

            return result

        # For larger numbers, use a simpler approach
        return str(num)  # Fallback to digits for very large numbers

    def phonemize(self, text: str, normalize: bool = True) -> str:
        """
        Convert Turkish text to phonemes.

        Args:
            text: Input Turkish text
            normalize: Whether to normalize text first (default: True)

        Returns:
            Phoneme sequence as string

        Example:
            >>> phonemizer = TurkishPhonemizer()
            >>> phonemizer.phonemize("Merhaba dünya")
            'm ɛ ɾ h a b a d y n j a'
        """
        if normalize:
            text = self.normalize_text(text)

        try:
            phonemes = self.backend.phonemize([text], strip=True)[0]
            return phonemes
        except Exception as e:
            raise RuntimeError(f"Phonemization failed: {e}")

    def phonemize_batch(self, texts: List[str], normalize: bool = True) -> List[str]:
        """
        Convert batch of Turkish texts to phonemes.

        Args:
            texts: List of input Turkish texts
            normalize: Whether to normalize texts first

        Returns:
            List of phoneme sequences
        """
        if normalize:
            texts = [self.normalize_text(text) for text in texts]

        try:
            phonemes = self.backend.phonemize(texts, strip=True)
            return phonemes
        except Exception as e:
            raise RuntimeError(f"Batch phonemization failed: {e}")

    def build_vocab_from_texts(self, texts: List[str], min_frequency: int = 1) -> Dict[str, int]:
        """
        Build phoneme vocabulary from list of texts.

        Args:
            texts: List of Turkish texts
            min_frequency: Minimum frequency for a phoneme to be included

        Returns:
            Phoneme vocabulary (phoneme → index mapping)
        """
        # Phonemize all texts
        all_phonemes = []
        for text in texts:
            phonemes = self.phonemize(text)
            # Split phonemes (espeak outputs space-separated phonemes)
            all_phonemes.extend(phonemes.split())

        # Count phoneme frequencies
        phoneme_counts = Counter(all_phonemes)

        # Filter by frequency
        filtered_phonemes = [
            phoneme for phoneme, count in phoneme_counts.items()
            if count >= min_frequency
        ]

        # Build vocabulary with special tokens
        vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        # Add phonemes (sorted for consistency)
        for idx, phoneme in enumerate(sorted(filtered_phonemes), start=4):
            vocab[phoneme] = idx

        self.phoneme_vocab = vocab
        self.index_to_phoneme = {v: k for k, v in vocab.items()}

        return vocab

    def encode(self, phonemes: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode phoneme string to indices.

        Args:
            phonemes: Space-separated phoneme string
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of phoneme indices
        """
        if not self.phoneme_vocab:
            raise ValueError("Vocabulary not built. Call build_vocab_from_texts() first.")

        # Split phonemes
        phoneme_list = phonemes.split()

        # Encode to indices
        indices = []

        if add_special_tokens:
            indices.append(self.phoneme_vocab[self.BOS_TOKEN])

        for phoneme in phoneme_list:
            indices.append(self.phoneme_vocab.get(phoneme, self.phoneme_vocab[self.UNK_TOKEN]))

        if add_special_tokens:
            indices.append(self.phoneme_vocab[self.EOS_TOKEN])

        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode phoneme indices to string.

        Args:
            indices: List of phoneme indices
            skip_special_tokens: Whether to skip special tokens (PAD, BOS, EOS)

        Returns:
            Space-separated phoneme string
        """
        if not self.index_to_phoneme:
            raise ValueError("Vocabulary not built.")

        special_tokens = {
            self.phoneme_vocab[self.PAD_TOKEN],
            self.phoneme_vocab[self.BOS_TOKEN],
            self.phoneme_vocab[self.EOS_TOKEN],
        }

        phonemes = []
        for idx in indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            phonemes.append(self.index_to_phoneme.get(idx, self.UNK_TOKEN))

        return " ".join(phonemes)

    def save_vocab(self, path: str) -> None:
        """
        Save phoneme vocabulary to JSON file.

        Args:
            path: Path to save vocabulary
        """
        vocab_path = Path(path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.phoneme_vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str) -> None:
        """
        Load phoneme vocabulary from JSON file.

        Args:
            path: Path to vocabulary file
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.phoneme_vocab = json.load(f)

        self.index_to_phoneme = {v: k for k, v in self.phoneme_vocab.items()}

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.phoneme_vocab)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TurkishPhonemizer(backend='espeak-ng', "
            f"vocab_size={self.get_vocab_size()})"
        )


# ============================================================================
# Utility functions
# ============================================================================

def create_phonemizer_from_config(config: Dict) -> TurkishPhonemizer:
    """
    Create TurkishPhonemizer from configuration dictionary.

    Args:
        config: Configuration dict with 'phonemizer' section

    Returns:
        Initialized TurkishPhonemizer
    """
    phonemizer_config = config.get('phonemizer', {})

    phonemizer = TurkishPhonemizer(
        language=phonemizer_config.get('language', 'tr'),
        preserve_punctuation=phonemizer_config.get('preserve_punctuation', True),
        with_stress=phonemizer_config.get('with_stress', True),
    )

    # Load vocab if path exists
    vocab_path = phonemizer_config.get('phoneme_vocab_path')
    if vocab_path and Path(vocab_path).exists():
        phonemizer.load_vocab(vocab_path)

    return phonemizer


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Test Turkish phonemizer
    phonemizer = TurkishPhonemizer()

    # Test sentences
    test_sentences = [
        "Merhaba, nasılsınız?",
        "Bugün hava çok güzel.",
        "123 kişi geldi.",
        "Dr. Ahmet'le görüştüm.",
        "Türkçe metin seslendirilmesi çalışıyor.",
    ]

    print("Turkish Phonemizer Test\n" + "="*50)

    for text in test_sentences:
        normalized = phonemizer.normalize_text(text)
        phonemes = phonemizer.phonemize(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print(f"Phonemes:   {phonemes}")
        print()

    # Build vocabulary
    print("\nBuilding vocabulary from test sentences...")
    vocab = phonemizer.build_vocab_from_texts(test_sentences)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample phonemes: {list(vocab.keys())[:10]}")

    # Test encoding/decoding
    print("\nTesting encoding/decoding...")
    test_text = "Merhaba dünya"
    phonemes = phonemizer.phonemize(test_text)
    encoded = phonemizer.encode(phonemes)
    decoded = phonemizer.decode(encoded)

    print(f"Text:     {test_text}")
    print(f"Phonemes: {phonemes}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
