from llama_index.embeddings.mistralai import MistralAIEmbedding
from dotenv import load_dotenv
import asyncio
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step

load_dotenv()

model_name="mistral-embed"

embed_model = MistralAIEmbedding(model_name=model_name)



import os
from typing import List, Optional

from llama_index.llms.text_generation_inference import (
    TextGenerationInference,
)

URL = "https://gc6fmqcfdtirbm5x.us-east-1.aws.endpoints.huggingface.cloud"
hf_llm = TextGenerationInference(
    model_url=URL, token=os.environ["HF_TOKEN"]
)

from llama_index.core import Settings

Settings.embed_model = embed_model

from llama_index.core import PromptTemplate

DEFAULT_RAG_PROMPT = PromptTemplate(
    template="""Use the provided context to answer the question. If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
)

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

class PrepEvent(Event):
    """Prep event (prepares for retrieval)."""
    pass

class RetrieveEvent(Event):
    """Retrieve event (gets retrieved nodes)."""

    retrieved_nodes: list[NodeWithScore]

class AugmentGenerateEvent(Event):
    """Query event. Queries given relevant text and search text."""
    relevant_text: str
    search_text: str


from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core import (
    VectorStoreIndex,
    Document,
    SummaryIndex,
)
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from IPython.display import Markdown, display
from llama_index.core.base.base_retriever import BaseRetriever

# First things first, we need to create a new class that subclasses Workflow.
# Each step, now, is a method (decorated by the @step decorator) which will take an Event and Context as input.
class OpenSourceRAG(Workflow):
    @step
    async def prepare_for_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> PrepEvent | None:
        """Prepare for retrieval."""

        model_url = "https://cx7s40y9qdd7zxhr.us-east-1.aws.endpoints.huggingface.cloud"

        query_str: str | None = ev.get("query_str")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        pinecone_index = pc.Index(os.environ["INDEX_NAME"])
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )
        index = VectorStoreIndex.from_vector_store(vector_store)

        llm = TextGenerationInference(
            model_url=model_url,
            token=os.environ["HF_TOKEN"],
            model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 "
        )
        await ctx.set("rag_pipeline", QueryPipeline(
            chain=[DEFAULT_RAG_PROMPT, llm]
        ))

        await ctx.set("llm", llm)
        await ctx.set("index", index)

        await ctx.set("query_str", query_str)
        await ctx.set("retriever_kwargs", retriever_kwargs)

        return PrepEvent()

    @step
    async def retrieve(
        self, ctx: Context, ev: PrepEvent
    ) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        retriever_kwargs = await ctx.get("retriever_kwargs")

        if query_str is None:
            return None

        index = await ctx.get("index", default=None)
        if not (index):
            raise ValueError(
                "Index and tavily tool must be constructed. Run with 'documents' and 'tavily_ai_apikey' params first."
            )

        retriever: BaseRetriever = index.as_retriever(
            **retriever_kwargs
        )
        result = retriever.retrieve(query_str)
        await ctx.set("query_str", query_str)
        return RetrieveEvent(retrieved_nodes=result)

    @step
    async def augment_and_generate(self, ctx: Context, ev: RetrieveEvent) -> StopEvent:
        """Get result with relevant text."""
        relevant_nodes = ev.retrieved_nodes
        relevant_text = "\n".join([node.get_content() for node in relevant_nodes])
        query_str = await ctx.get("query_str")

        relevancy_pipeline = await ctx.get("rag_pipeline")

        relevancy = relevancy_pipeline.run(
                context=relevant_text, question=query_str
        )

        return StopEvent(result=relevancy.message.content)
    
open_source_rag_wf = OpenSourceRAG()
    
async def main():
    print(await open_source_rag_wf.run(query_str="What state is the complaint filed in?"))

if __name__ == "__main__":
    asyncio.run(main())