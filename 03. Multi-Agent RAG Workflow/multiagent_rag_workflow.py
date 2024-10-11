import os
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import PromptTemplate
from llama_index.llms.text_generation_inference import (
    TextGenerationInference,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from dotenv import load_dotenv
from typing import Optional, List, Callable
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core import (
    VectorStoreIndex,
    Document,
    SummaryIndex,
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from IPython.display import Markdown, display
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from colorama import Fore, Back, Style

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)

### --- INITS --- ###

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

embed_model = MistralAIEmbedding(model_name="mistral-embed")
Settings.embed_model = embed_model

pinecone_index = pc.Index(os.environ["INDEX_NAME"])
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
)
storage_context = StorageContext.from_defaults(
  vector_store=vector_store
)
vector_store_index = VectorStoreIndex.from_vector_store(vector_store)

DEFAULT_RAG_PROMPT = PromptTemplate(
    template="""Use the provided context to answer the question. If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
)

### ---!! RAG QueryEngine Tool !!--- ###

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: TextGenerationInference
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)
    
pincecone_retriever = vector_store_index.as_retriever()
synthesizer = get_response_synthesizer(response_mode="compact")

hf_llm = TextGenerationInference(
    model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 ",
    model_url=os.environ["HF_LLM_URL"], 
    token=os.environ["HF_TOKEN"]
)

rag_query_engine = RAGStringQueryEngine(
    retriever=pincecone_retriever,
    response_synthesizer=synthesizer,
    llm=hf_llm,
    qa_prompt=qa_prompt,
)

rag_tool = QueryEngineTool.from_defaults(
    rag_query_engine, name="ragtool", description="Useful for when you want to more about Elon Musk's complaint to OpenAI"
)
    
### ---!! MULTI-AGENT WORKFLOW !!--- ###

### --- EVENTS --- ####

class InitializeEvent(Event):
    pass

class InterfaceAgentEvent(Event):
    request: Optional[str] = None
    just_completed: Optional[str] = None
    need_help: Optional[bool] = None

class OrchestratorEvent(Event):
    request: str

class WebSearchEvent(Event):
    request: str

class RAGSearchEvent(Event):
    request: str

### --- WORKFLOW --- ###

class MultiAgentRag(Workflow):
    @step(pass_context=True)
    async def initialize(self, ctx: Context, ev: InitializeEvent) -> InterfaceAgentEvent:
        ctx.data["initialized"] = None
        ctx.data["success"] = None
        ctx.data["redirecting"] = None
        ctx.data["overall_request"] = None

        ctx.data["llm"] = OpenAI(model="gpt-4o",temperature=0.4)

        return InterfaceAgentEvent()

    @step(pass_context=True)
    async def interface_agent(self, ctx: Context, ev: InterfaceAgentEvent | StartEvent) -> InitializeEvent | StopEvent | OrchestratorEvent:
        if ("initialized" not in ctx.data):
            return InitializeEvent()
        
        if ("InterfaceAgent" not in ctx.data):
            system_prompt = (f"""
                You are a helpful assistant that is helping a user navigate a Elon Musk's various beefs with other notable people throughout the years.
                Your job is to ask the user questions to figure out what they want to do, and give them the available things they can do.
                That includes
                * Searching the web for information
                * Consulting the OpenAI Lawsuit for information
                You should start by listing the things you can help them do.            
            """)

            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[],
                llm=ctx.data["llm"],
                allow_parallel_tool_calls=False,
                system_prompt=system_prompt
            )
            ctx.data["InterfaceAgent"] = agent_worker.as_agent()       

        interface_agent = ctx.data["InterfaceAgent"]
        if ctx.data["overall_request"] is not None:
            print("There's an overall request in progress, it's ", ctx.data["overall_request"])
            last_request = ctx.data["overall_request"]
            ctx.data["overall_request"] = None
            return OrchestratorEvent(request=last_request)
        elif (ev.just_completed is not None):
            response = interface_agent.chat(f"FYI, the user has just completed the task: {ev.just_completed}")
        elif (ev.need_help):
            print("The previous process needs help with ", ev.request)
            return OrchestratorEvent(request=ev.request)
        else:
            response = interface_agent.chat("Hello!")

        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)
        user_msg_str = input("> ").strip()
        return OrchestratorEvent(request=user_msg_str)
    
    @step(pass_context=True)
    async def orchestrator(self, ctx: Context, ev: OrchestratorEvent) -> InterfaceAgentEvent | WebSearchEvent | RAGSearchEvent | StopEvent:

        print(f"Orchestrator received request: {ev.request}")

        def emit_web_lookup() -> bool:
            """Call this if the user wants to look up information about Elon Musk and another person's beef."""      
            print("__emitted: WebSearchEvent")      
            self.send_event(WebSearchEvent(request=ev.request))
            return True

        def emit_rag_lookup() -> bool:
            """Call this if the user wants to learn about Elon Musk's complaint to OpenAI in RAG"""
            print("__emitted: RAGSearchEvent")
            self.send_event(RAGSearchEvent(request=ev.request))
            return True

        def emit_interface_agent() -> bool:
            """Call this if the user wants to do something else or you can't figure out what they want to do."""
            print("__emitted: interface")
            self.send_event(InterfaceAgentEvent(request=ev.request))
            return True

        def emit_stop() -> bool:
            """Call this if the user wants to stop or exit the system."""
            print("__emitted: stop")
            self.send_event(StopEvent())
            return True

        tools = [
            FunctionTool.from_defaults(fn=emit_web_lookup),
            FunctionTool.from_defaults(fn=emit_rag_lookup),
            FunctionTool.from_defaults(fn=emit_interface_agent),
            FunctionTool.from_defaults(fn=emit_stop)
        ]
        
        system_prompt = (f"""
            You are on orchestration agent.
            Your job is to decide which agent to run based on the current state of the user and what they've asked to do. 
            You run an agent by calling the appropriate tool for that agent.
            You do not need to call more than one tool.
            You do not need to figure out dependencies between agents; the agents will handle that themselves.
                            
            If you did not call any tools, return the string "FAILED" without quotes and nothing else.
        """)

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=ctx.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt
        )
        ctx.data["orchestrator"] = agent_worker.as_agent()        
        
        orchestrator = ctx.data["orchestrator"]
        response = str(orchestrator.chat(ev.request))

        if response == "FAILED":
            print("Orchestration agent failed to return a valid speaker; try again")
            return OrchestratorEvent(request=ev.request)
        
    @step(pass_context=True)
    async def web_search(self, ctx: Context, ev: WebSearchEvent) -> InterfaceAgentEvent:

        print(f"WebSearch received request: {ev.request}")

        if ("web_search_agent" not in ctx.data):
            taviliy_tool = TavilyToolSpec(api_key=os.environ["TAVILY_API_KEY"])


            system_prompt = (f"""
                You are a helpful assistant that is looking up information on Elon Musk and his various beefs with other notable people throughout the years.
                Once you have retrieved information about the beef, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than look up information on Elon Musk and his various beefs, call the tool "need_help" to signal some other agent should help.
            """)

            ctx.data["web_search_agent"] = BaseAgent(
                name="Web Search Agent",
                parent=self,
                tools=taviliy_tool.to_tool_list(),
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=WebSearchEvent
            )

        return ctx.data["web_search_agent"].handle_event(ev)
    
    @step(pass_context=True)
    async def rag_lookup(self, ctx: Context, ev: RAGSearchEvent) -> InterfaceAgentEvent:
            
        print(f"RAG received request: {ev.request}")

        if ("rag_search_agent" not in ctx.data):
            rag_tool = QueryEngineTool.from_defaults(
                rag_query_engine, name="ragtool", description="Useful for when you want to more about Elon Musk's complaint to OpenAI"
            )


            system_prompt = (f"""
                You are a helpful assistant that is looking up information on Elon Musk and his complaint/lawsuit with OpenAI.
                Once you have retrieved information about the complaint/lawsuit, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than look up information on Elon Musk, call the tool "need_help" to signal some other agent should help.
            """)

            ctx.data["rag_search_agent"] = BaseAgent(
                name="RAG Search Agent",
                parent=self,
                tools=[rag_tool],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=RAGSearchEvent
            )

        return ctx.data["rag_search_agent"].handle_event(ev)

class BaseAgent():
    name: str
    parent: Workflow
    tools: list[FunctionTool]
    system_prompt: str
    context: Context
    current_event: Event
    trigger_event: Event

    def __init__(
            self,
            parent: Workflow,
            tools: List[Callable], 
            system_prompt: str, 
            trigger_event: Event,
            context: Context,
            name: str,
        ):
        self.name = name
        self.parent = parent
        self.context = context
        self.system_prompt = system_prompt
        self.context.data["redirecting"] = False
        self.trigger_event = trigger_event

        # set up the tools including the ones everybody gets
        def done() -> None:
            """When you complete your task, call this tool."""
            print(f"{self.name} is complete")
            self.context.data["redirecting"] = True
            parent.send_event(InterfaceAgentEvent(just_completed=self.name))

        def need_help() -> None:
            """If the user asks to do something you don't know how to do, call this."""
            print(f"{self.name} needs help")
            self.context.data["redirecting"] = True
            parent.send_event(InterfaceAgentEvent(request=self.current_event.request,need_help=True))

        self.tools = [
            FunctionTool.from_defaults(fn=done),
            FunctionTool.from_defaults(fn=need_help)
        ]
        for t in tools:
            self.tools.append(t)

        agent_worker = FunctionCallingAgentWorker.from_tools(
            self.tools,
            llm=self.context.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=self.system_prompt
        )
        self.agent = agent_worker.as_agent()        

    def handle_event(self, ev: Event):
        self.current_event = ev

        response = str(self.agent.chat(ev.request))
        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        # if they're sending us elsewhere we're done here
        if self.context.data["redirecting"]:
            self.context.data["redirecting"] = False
            return None

        # otherwise, get some user input and then loop
        user_msg_str = input("> ").strip()
        return self.trigger_event(request=user_msg_str)
    

draw_all_possible_flows(MultiAgentRag,filename="multi-agent-rag.html")

async def main():
    c = MultiAgentRag(timeout=1200, verbose=True)
    result = await c.run()
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())