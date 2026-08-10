from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    provider: str
    model: str
    model_version: str
    dimensions: int


@dataclass(frozen=True)
class EmbeddingIdentitySpec:
    profile_name: str
    provider: str
    model: str
    model_version: str
    dimensions: int

    @property
    def fingerprint(self) -> str:
        payload = {
            "profile_name": self.profile_name,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
        }

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            canonical.encode("utf-8"),
        ).hexdigest()