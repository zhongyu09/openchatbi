"""Text segmentation utility with jieba support."""

import re
import string
import sys

# Try to import jieba, fallback to None if not available
# Note: jieba is not compatible with Python 3.12+
_jieba_available = False
if sys.version_info < (3, 12):
    try:
        import jieba

        _jieba_available = True
    except ImportError:
        _jieba_available = False


class TextSegmenter:
    """A text segmenter that uses jieba for Chinese text and simple splitting for others.

    This segmenter tries to use jieba for better Chinese word segmentation.
    If jieba is not available or Python version is 3.12+, it falls back to simple
    punctuation/whitespace splitting.

    Note: jieba is not compatible with Python 3.12+, so simple segmentation will be
    used on Python 3.12 and higher versions.
    """

    def __init__(self, use_jieba: bool = True):
        """Initialize the text segmenter.

        Args:
            use_jieba: Whether to use jieba for Chinese text segmentation.
                Defaults to True. Will automatically fall back to simple
                segmentation if jieba is not available.
        """
        self.use_jieba = use_jieba and _jieba_available

        # Include both English and Chinese punctuation
        chinese_punctuation = "，。！？；：" "''（）【】《》〈〉「」『』〔〕"
        all_separators = string.punctuation + chinese_punctuation + " \t\n\r"
        # Create regex pattern to split on any separator
        self.split_pattern = "[" + re.escape(all_separators) + "]+"

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Check if text contains Chinese characters.

        Args:
            text: Input text to check

        Returns:
            True if text contains Chinese characters, False otherwise
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _simple_cut(self, text: str) -> list[str]:
        """Simple segmentation by splitting on punctuation and whitespace.

        Args:
            text: Input text to be segmented

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Split by separators and filter empty strings
        tokens = re.split(self.split_pattern, text)
        return [token for token in tokens if token.strip()]

    def cut(self, text: str) -> list[str]:
        """Segment text into tokens.

        For Chinese text with jieba available, uses jieba for word segmentation.
        Otherwise, splits by punctuation and whitespace.

        Args:
            text: Input text to be segmented

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Use jieba for Chinese text if available
        if self.use_jieba and self._contains_chinese(text):
            return list(jieba.cut(text))

        # Fall back to simple segmentation
        return self._simple_cut(text)


class SimpleSegmenter:
    """A simple text segmenter that splits text by punctuation and whitespace.

    This is a lightweight text segmentation tool that provides basic
    functionality without external dependencies.

    Note: This class is kept for backward compatibility. Consider using
    TextSegmenter instead for better Chinese text support.
    """

    def __init__(self):
        # Include both English and Chinese punctuation
        chinese_punctuation = "，。！？；：" "''（）【】《》〈〉「」『』〔〕"
        all_separators = string.punctuation + chinese_punctuation + " \t\n\r"
        # Create regex pattern to split on any separator
        self.split_pattern = "[" + re.escape(all_separators) + "]+"

    def cut(self, text: str) -> list[str]:
        """Segment text into tokens by splitting on punctuation and whitespace.

        Args:
            text: Input text to be segmented

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Split by separators and filter empty strings
        tokens = re.split(self.split_pattern, text)
        return [token for token in tokens if token.strip()]


# Global instance - use TextSegmenter with jieba support
_segmenter = TextSegmenter()
