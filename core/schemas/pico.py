from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator

class PICO(BaseModel):
    Population: str = Field(..., min_length=1)
    Intervention: str = Field(..., min_length=1)
    Comparator: Optional[str] = None
    Outcomes: List[str] = Field(default_factory=list)

    @field_validator("Outcomes", mode="before")
    @classmethod
    def outcomes_list_or_str(cls, v):
        if v is None: return []
        if isinstance(v, str): return [v]
        return v

class QuoteBacks(BaseModel):
    Population: Optional[str] = None
    Intervention: Optional[str] = None
    Comparator: Optional[str] = None
    Outcomes: Dict[str, str] = Field(default_factory=dict)

class AugmentedField(BaseModel):
    value: str
    synonyms: List[str] = Field(default_factory=list)

class AugmentedPICO(BaseModel):
    Population: AugmentedField
    Intervention: AugmentedField
    Comparator: Optional[AugmentedField] = None
    Outcomes: List[AugmentedField] = Field(default_factory=list)