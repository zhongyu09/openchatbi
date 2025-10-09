"""Simple text segmentation utility."""

import re
import string
from typing import List


class SimpleSegmenter:
    """A simple text segmenter that splits text by punctuation and whitespace.

    This is a lightweight text segmentation tool that provides basic
    functionality without external dependencies.
    """

    def __init__(self):
        # Include both English and Chinese punctuation
        chinese_punctuation = "，。！？；：" "''（）【】《》〈〉「」『』〔〕"
        all_separators = string.punctuation + chinese_punctuation + " \t\n\r"
        # Create regex pattern to split on any separator
        self.split_pattern = "[" + re.escape(all_separators) + "]+"

    def cut(self, text: str) -> List[str]:
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


# Global instance
_segmenter = SimpleSegmenter()
