"""Easy Data Augmentation (Wei and Zou, EMNLP 2019) with a small Indonesian synonym list."""

from __future__ import annotations

from typing import List, Optional

from src.augment.base import Augmenter




INDONESIAN_SYNONYMS: dict[str, List[str]] = {
    "bagus": ["baik", "hebat", "keren", "mantap"],
    "buruk": ["jelek", "tidak baik", "payah"],
    "cepat": ["lekas", "segera", "gesit"],
    "lambat": ["pelan", "lelet"],
    "besar": ["agung", "raksasa", "akbar"],
    "kecil": ["mungil", "mini"],
    "sangat": ["amat", "begitu", "betul-betul"],
    "selalu": ["senantiasa", "kerap"],
    "sering": ["kerap", "acap kali"],
    "kadang": ["sesekali", "terkadang"],
    "mungkin": ["barangkali", "boleh jadi"],
    "pasti": ["tentu", "tentunya", "niscaya"],
    "menemukan": ["mendapati", "menjumpai"],
    "membuat": ["menciptakan", "mengerjakan"],
    "berbicara": ["bertutur", "berkata"],
    "menjelaskan": ["menerangkan", "memaparkan"],
    "menggunakan": ["memakai", "memanfaatkan"],
    "penting": ["krusial", "vital", "utama"],
    "baru": ["anyar", "mutakhir"],
    "lama": ["usang", "kuno"],
    "tinggi": ["jangkung", "tegak"],
    "rendah": ["minim", "kurang"],
    "banyak": ["beragam", "berlimpah", "berbagai"],
    "sedikit": ["minim", "tipis"],
    "kuat": ["tangguh", "perkasa"],
    "lemah": ["lemas", "loyo"],
    "tahu": ["paham", "mengerti"],
    "mengetahui": ["memahami", "mengerti"],
    "memberi": ["menyerahkan", "memberikan"],
    "menerima": ["mendapat", "memperoleh"],
    "memulai": ["mengawali", "merintis"],
    "selesai": ["rampung", "tuntas", "kelar"],
    "mulai": ["awali", "rintis"],
    "kembali": ["balik", "pulang"],
    "pergi": ["berangkat", "bertolak"],
    "datang": ["tiba", "muncul"],
    "lihat": ["pandang", "tatap", "amati"],
    "melihat": ["memandang", "mengamati", "menatap"],
    "kata": ["ujar", "ucap"],
    "berita": ["kabar", "informasi", "warta"],
    "negara": ["bangsa", "tanah air"],
    "rakyat": ["warga", "masyarakat"],
    "pemerintah": ["pemerintahan", "kekuasaan"],
    "uang": ["duit", "rupiah"],
    "rumah": ["hunian", "tempat tinggal"],
    "mobil": ["kendaraan", "mobil pribadi"],
    "anak": ["bocah", "buah hati"],
    "orang": ["insan", "individu", "manusia"],
    "wanita": ["perempuan"],
    "pria": ["laki-laki", "lelaki"],
    "hari": ["waktu"],
    "tahun": ["periode"],
    "viral": ["ramai", "menyebar luas"],
    "hoax": ["bohong", "tipuan"],
    "fakta": ["kenyataan", "realitas"],
    "klaim": ["pernyataan", "dakwaan"],
    "bukti": ["fakta", "petunjuk"],
}


def _tokenize(text: str) -> List[str]:
    return text.split()


def _detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


class EDAAugmenter(Augmenter):
    """Easy Data Augmentation: synonym replacement, swap, delete, or insert."""

    def __init__(
        self,
        op: str = "synonym",
        alpha: float = 0.15,
        seed: Optional[int] = None,
        synonyms: Optional[dict] = None,
    ) -> None:
        super().__init__(seed=seed)
        assert op in {"synonym", "swap", "delete", "insert"}, f"unknown op: {op}"
        self.op = op
        self.alpha = float(alpha)
        self.name = op
        self.synonyms = synonyms if synonyms is not None else INDONESIAN_SYNONYMS



    def _synonym_replace(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return tokens
        n = max(1, int(round(self.alpha * len(tokens))))
        candidate_idx = [i for i, t in enumerate(tokens) if t.lower() in self.synonyms]
        if not candidate_idx:
            return tokens
        self.rng.shuffle(candidate_idx)
        new = list(tokens)
        replaced = 0
        for i in candidate_idx:
            options = self.synonyms.get(new[i].lower())
            if not options:
                continue
            new[i] = self.rng.choice(options)
            replaced += 1
            if replaced >= n:
                break
        return new

    def _swap(self, tokens: List[str]) -> List[str]:
        if len(tokens) < 2:
            return tokens
        n = max(1, int(round(self.alpha * len(tokens))))
        new = list(tokens)
        for _ in range(n):
            i, j = self.rng.sample(range(len(new)), 2)
            new[i], new[j] = new[j], new[i]
        return new

    def _delete(self, tokens: List[str]) -> List[str]:
        if len(tokens) <= 1:
            return tokens
        kept = [t for t in tokens if self.rng.random() >= self.alpha]
        if not kept:
            kept = [self.rng.choice(tokens)]
        return kept

    def _insert(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return tokens
        candidate_words = [t for t in tokens if t.lower() in self.synonyms]
        if not candidate_words:
            return tokens
        n = max(1, int(round(self.alpha * len(tokens))))
        new = list(tokens)
        for _ in range(n):
            word = self.rng.choice(candidate_words)
            syn = self.rng.choice(self.synonyms[word.lower()])
            pos = self.rng.randint(0, len(new))
            new.insert(pos, syn)
        return new

    def augment(self, text: str) -> str:
        tokens = _tokenize(text)
        if not tokens:
            return text
        op_fn = {
            "synonym": self._synonym_replace,
            "swap": self._swap,
            "delete": self._delete,
            "insert": self._insert,
        }[self.op]
        return _detokenize(op_fn(tokens))
