from __future__ import annotations
from typing import Optional

from agents.agent import Agent
from agents.utils.types import GenParams
from agents.utils.clean import clean_reply
from agents.backends.vertex_predict import VertexPredictBackend
from agents.backends.vertex_generative import VertexGenerativeBackend
# from agents.backends.openai_chat import OpenAIChatBackend  # enable when needed


def build_vertex_agent(
    *,
    project_id: str,
    location: str,
    endpoint_id: str,
    dedicated_dns_or_predict_url: str,
    model: Optional[str] = None,  # informational; not required by predict
    max_tokens: int = 10000,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop: Optional[list[str]] = None,
) -> Agent:
    backend = VertexPredictBackend(
        project_id=project_id,
        location=location,
        endpoint_id=endpoint_id,
        dedicated_dns_or_predict_url=dedicated_dns_or_predict_url,
        model=model,
    )
    params = GenParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop,
    )
    # Cleaner is provider-agnostic; Vertex backend has default_stop to suppress markers.
    return Agent(backend, default_params=params, cleaner=clean_reply)


def build_vertex_gemini_agent(
    *,
    project_id: str,
    location: str = "us-central1",
    model: str = "gemini-2.5-flash",
    max_tokens: int = 4000,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop: Optional[list[str]] = None,
) -> Agent:
    """
    Build an Agent backed by Vertex AI's Gemini family (GenerativeModel API).

    Use this for tasks where you need strong instruction-following and clean
    structured-output discipline (e.g., JSON term extraction). For private
    deployed models (medgemma etc.), keep using build_vertex_agent.

    Auth: Application Default Credentials.
        gcloud auth application-default login
    """
    backend = VertexGenerativeBackend(
        project_id=project_id,
        location=location,
        model=model,
    )
    params = GenParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop,
    )
    return Agent(backend, default_params=params, cleaner=clean_reply)


# Example future builder
# def build_openai_agent(
#     *,
#     api_key: str,
#     model: str = "gpt-5",
#     base_url: str = "https://api.openai.com/v1",
#     max_tokens: int = 1024,
#     temperature: float = 0.2,
#     top_p: Optional[float] = None,
#     stop: Optional[list[str]] = None,
# ) -> Agent:
#     backend = OpenAIChatBackend(api_key=api_key, model=model, base_url=base_url)
#     params = GenParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
#     return Agent(backend, default_params=params)  # same cleaner works
