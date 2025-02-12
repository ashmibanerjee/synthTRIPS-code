import os
import sys

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

model = get_registry().get("sentence-transformers").create()


class KnowledgeBase(LanceModel):
    """
    
    Schema definition for the Knowledge Base.

    """
    city: str = model.SourceField()
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()
