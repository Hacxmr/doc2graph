"""LangChain adapters for relation building: embeddings and the optional LLM classifier.

    classifier = langchain_classifier(init_chat_model("<model>"))
    embed = langchain_embedder(embeddings)          # any LangChain ``Embeddings``
    result = build_document_graph(pages, typed=True, embed=embed, classifier=classifier)

LangChain only labels candidate pairs and embeds text; edges are created and
checked by ``data/relations.py``. Needs ``uv sync --extra langchain`` (or one of
the ``huggingface`` / ``local-embeddings`` extras).
"""

import os
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class EdgeDecision(BaseModel):
    """Structured answer the chat model must give for one candidate pair."""

    relation: Literal["CONTINUES", "SAME_TABLE", "REFERS_TO", "SIMILAR_TO", "NONE"]
    confidence: float = Field(ge=0.0, le=1.0, description="how sure the answer is, 0 to 1")
    evidence: str = Field(description="one short sentence quoting the supporting text")


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You label the relation between two blocks of one document. Allowed relations: "
            "{allowed}, or NONE. CONTINUES: B continues the sentence or paragraph of A. SAME_TABLE: "
            "B continues the table of A. REFERS_TO: one block names the figure or table the other "
            "carries. SIMILAR_TO: the blocks are about the same thing but none of the others applies. "
            "Answer with a relation only when the text itself supports it; otherwise answer NONE.",
        ),
        (
            "human",
            "Block A (page {source_page}, {source_type}):\n{source_text}\n\n"
            "Block B (page {target_page}, {target_type}):\n{target_text}\n\n"
            "Rule evidence: {evidence}\n{format_instructions}",
        ),
    ]
)


def langchain_classifier(model):
    """Turn a LangChain chat model into a ``classifier`` for ``build_document_graph``.

    Uses ``prompt | model.with_structured_output(EdgeDecision)``. A model without
    structured-output support falls back to a JSON-instructed prompt parsed into
    the same schema. Only the two blocks are sent, never the document. Output is
    validated downstream by ``relations.validate_decision``.
    """
    try:
        chain, instructions = _PROMPT | model.with_structured_output(EdgeDecision), ""
    except NotImplementedError:
        parser = PydanticOutputParser(pydantic_object=EdgeDecision)
        chain, instructions = _PROMPT | model | parser, parser.get_format_instructions()

    def classify(candidate):
        answer = chain.invoke(
            {
                "allowed": ", ".join(candidate["allowed"]),
                "source_page": candidate["source"]["page"] + 1,
                "source_type": candidate["source_type"],
                "source_text": candidate["source_text"],
                "target_page": candidate["target"]["page"] + 1,
                "target_type": candidate["target_type"],
                "target_text": candidate["target_text"],
                "evidence": candidate["evidence"],
                "format_instructions": instructions,
            }
        )
        return answer if isinstance(answer, dict) else dict(answer)

    return classify


def huggingface_models(
    llm="meta-llama/Llama-3.1-8B-Instruct",
    embeddings="sentence-transformers/all-MiniLM-L6-v2",
    token=None,
    provider=None,
):
    """Hugging Face Inference API chat model and embeddings, ready for the adapters.

    The token comes from ``token`` or the ``HF_TOKEN`` environment variable.
    Needs ``uv sync --extra huggingface``. Returns ``(chat_model, embeddings)``.
    """
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN (a Hugging Face access token) or pass token=")
    extra = {"provider": provider} if provider else {}
    endpoint = HuggingFaceEndpoint(
        repo_id=llm,
        huggingfacehub_api_token=token,
        task="text-generation",
        max_new_tokens=256,
        temperature=0.01,  # near-deterministic labelling
        **extra,
    )
    chat = ChatHuggingFace(llm=endpoint)
    return chat, HuggingFaceEndpointEmbeddings(model=embeddings, huggingfacehub_api_token=token)


def langchain_embedder(embeddings):
    """Use a LangChain ``Embeddings`` object as the ``embed`` callable."""
    return embeddings.embed_documents


def local_embeddings(model="sentence-transformers/all-MiniLM-L6-v2"):
    """Embeddings computed on this machine (no token; downloads the model once).

    Needs ``uv sync --extra local-embeddings``. Pass the result to ``langchain_embedder``.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model)
