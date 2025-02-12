import json
from vertexai.generative_models import GenerationResponse, GenerationConfig
from typing import List, Dict, Any


def convert_to_serializable(d):
    """
    Recursively converts dictionary objects into JSON serializable types.
    """
    if isinstance(d, dict):
        return {k: convert_to_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_serializable(v) for v in d]
    elif isinstance(d, (int, float, str, bool)):
        return d  # primitive types are serializable as-is
    else:
        # Try to convert complex objects into serializable ones
        return str(d)  # This can be modified depending on your use case


def generation_response_to_json(response: GenerationResponse) -> Dict[str, Any]:
    """
    Converts a vertexai.generative_models.GenerationResponse object into a JSON-serializable dictionary.

    Args:
        response: The GenerationResponse object to convert.

    Returns:
        A dictionary representation of the GenerationResponse object suitable for JSON serialization.
    """
    print("Citation metadata: ", response.candidates[0].citation_metadata)
    response_dict = {
        "candidates": [
            {

                "content": response.text,
                "index": candidate.index,
                "avg_log_prob": candidate.avg_logprobs,
                "citation_metadata": [
                    {
                        "citation_sources": [
                            {
                                "end_index": citation.end_index,
                                "start_index": citation.start_index,
                                "uri": citation.uri,
                            }
                        ]
                    }
                    for citation in candidate.citation_metadata.citations
                ],
                "grounding_chunks": [
                    {
                        "uri": g.web.uri,
                        "title": g.web.title,
                    }
                    for g in candidate.grounding_metadata.grounding_chunks
                ],
                "grounding_supports": [
                    {
                        "grounding_chunk_index": g.grounding_chunk_indices[0],
                        "confidence_score": g.confidence_scores[0],
                        "text": g.segment.text,
                        "start_index": g.segment.start_index,
                        "end_index": g.segment.end_index,
                    }
                    for g in candidate.grounding_metadata.grounding_supports
                ],
            }
            for candidate in response.candidates
        ],
    }
    serializable_dict = convert_to_serializable(response_dict)

    return serializable_dict
