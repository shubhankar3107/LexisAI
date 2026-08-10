from enum import Enum


class VectorIndexStatus(str, Enum):
    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"