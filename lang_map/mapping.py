import random
import json
from typing import List, Union, Tuple, Dict

class PrivateUnicodeMapper:
    """
    Map characters from specified Unicode ranges to Private Use Areas (PUA, SPUA-A, SPUA-B)
    and support bi-directional string conversion.
    """

    def __init__(self, old_areas_list: List[Union[Tuple[int, int], int]], seed: int = 42):
        """
        Initialize the mapper and generate a random mapping.

        Args:
            old_areas_list: List of Unicode ranges (start, end) or single code points.
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.old_codes = self._collect_codes(old_areas_list)
        self.private_area = self._build_private_area()
        self._check_capacity()
        self.mapping, self.reverse_mapping = self._build_mapping()

    @classmethod
    def from_mapping_dict(cls, mapping_file: Union[Dict[str, int], str]):
        """Create an instance from an existing mapping dictionary or JSON file."""
        if isinstance(mapping_file, str):
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping_dict = json.load(f)
        else:
            mapping_dict = mapping_file

        mapping = {ord(k): v for k, v in mapping_dict.items()}
        reverse_mapping = {v: k for k, v in mapping.items()}
        instance = cls.__new__(cls)
        instance.mapping = mapping
        instance.reverse_mapping = reverse_mapping
        return instance

    def save_mapping(self, filepath: str):
        """Save the current mapping to a JSON file. old to new"""
        mapping_dict = {chr(k): v for k, v in self.mapping.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

    def get_vocab(self) -> Dict[str, int]:
        private_chars = list(self.reverse_mapping.keys())
        vocab = {chr(c): i for i, c in enumerate(private_chars)}

        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    def _collect_codes(self, old_areas_list):
        """Collect all Unicode code points from ranges or single points."""
        codes = []
        for area in old_areas_list:
            if isinstance(area, tuple):
                start, end = area
                codes.extend(range(start, end + 1))
            elif isinstance(area, int):
                codes.append(area)
            else:
                raise ValueError(f"Invalid area: {area}")
        return codes

    def _build_private_area(self):
        """Return list of all private area code points (PUA + SPUA-A + SPUA-B)."""
        pua_basic = range(0xE000, 0xF8FF + 1)
        pua_a = range(0xF0000, 0xFFFFD + 1)
        pua_b = range(0x100000, 0x10FFFD + 1)
        return list(pua_basic) + list(pua_a) + list(pua_b)

    def _check_capacity(self):
        """Check if enough private area slots exist for all original code points."""
        if len(self.old_codes) > len(self.private_area):
            raise ValueError(
                f"Number of original code points ({len(self.old_codes)}) exceeds available private area "
                f"({len(self.private_area)}). Reduce old_areas or split mapping."
            )

    def _build_mapping(self):
        """Randomly assign old codes to private area and create reverse mapping."""
        random.seed(self.seed)
        mapped_private = random.sample(self.private_area, len(self.old_codes))
        mapping = {orig: priv for orig, priv in zip(self.old_codes, mapped_private)}
        reverse_mapping = {v: k for k, v in mapping.items()}
        return mapping, reverse_mapping

    def map_string(self, text: str) -> str:
        """Map a string's characters to private area characters."""
        return "".join(chr(self.mapping.get(ord(c), ord(c))) for c in text)

    def unmap_string(self, text: str) -> str:
        """Decode a private area string back to original characters."""
        return "".join(chr(self.reverse_mapping.get(ord(c), ord(c))) for c in text)


if __name__ == "__main__":
    li = [
        '3002', 'FF1F', 'FF01', '3010', '3011', 'FF0C', '3001', 'FF1B',
        'FF1A', '300C', '300D', '300E', '300F', '2019', '201C', '201D',
        '2018', 'FF08', 'FF09', '3014', '3015', '2026', '2013', 'FF0E',
        '2014', '300A', '300B', '3008', '3009'
    ]
    punct_codes = [int(x, 16) for x in li]

    MAPPER = PrivateUnicodeMapper(
        old_areas_list=[(0x4E00, 0x9FFF), *punct_codes] # can add other zh ranges
    )
    MAPPER.get_vocab()
    MAPPER.save_mapping("mapping.json")

    zh = "就是因为有这些人认同我……所以我才能够……无论是有妖狐在我的体内，或是被村子里的人们以冷漠的眼光看待，我都不觉得难过了……因为我……已经不是孤单一人了！"
    s = MAPPER.map_string(zh)
    print(s)

    zh_back = MAPPER.unmap_string(s)
    print(zh_back, zh_back==zh)

    NEW_MAPPER = PrivateUnicodeMapper.from_mapping_dict(mapping_file="mapping.json")
    ns = NEW_MAPPER.map_string(zh)
    print(ns, s == ns)

    zh_back_n = NEW_MAPPER.unmap_string(ns)
    print(zh_back_n, zh_back_n == zh)