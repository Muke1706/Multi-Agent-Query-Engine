import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch # (Fixed)

# --- 1. Load API Keys ---
load_dotenv()

# --- 2. Define the State ---
# (NEW) We add 'final_answer' to hold the clean, final response
class AgentState(TypedDict):
    question: str
    doc_answer: str
    web_answer: str
    route: str
    final_answer: str # (NEW) This will hold the synthesizer's output

# --- 3. Set up Agents and Tools (No changes) ---
llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./db", 
    embedding_function=embeddings,
    collection_name="documents"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
rag_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Use the following context to answer the user's question.
    Context: {context}
    User's Question: {question}
    Answer:
    """
)
web_search_tool = TavilySearch(max_results=3) # (Fixed)

# --- 4. Define Agent Nodes (No changes) ---
def document_agent_node(state: AgentState):
    """This is the DocumentAgent node. It performs RAG."""
    print("--- Executing DocumentAgent Node ---")
    question = state['question']
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(question)
        print(f"DocumentAgent Answer: {answer[:200]}...")
        return {"doc_answer": answer}
    except Exception as e:
        print(f"Error in DocumentAgent: {e}")
        return {"doc_answer": f"Error: {e}"}

def web_search_node(state: AgentState):
    """This is the WebSearchAgent node. It searches the internet."""
    print("--- Executing WebSearchAgent Node ---")
    question = state['question']
    try:
        results_dict = web_search_tool.invoke({"query": question})
        snippets_list = results_dict.get("results", [])
        answer = "\n\n".join([snippet["content"] for snippet in snippets_list])
        print(f"WebSearchAgent Answer: {answer[:200]}...")
        return {"web_answer": answer}
    except Exception as e:
        print(f"Error in WebSearchAgent: {e}")
        return {"web_answer": f"Error: {e}"}

# --- 5. Define the Router Node (No changes) ---
router_prompt_template = """
You are an expert router. Your job is to classify a user's question to determine the best source of information.

Your first priority is to check if the question can be answered by the private documents.
The private documents contain information about:
- Project Management principles
- Project stakeholders
- Project management methodologies
- Leadership and team management in projects

Classify the question into one of two categories:
1.  'document_search': If the question is clearly about project management, stakeholders, or related topics covered in the private documents.
2.  'web_search': For all other questions (e.g., real-time information, weather, news, public figures, general knowledge not related to the documents).

Output *only* the single word 'document_search' or 'web_search'.

Question: {question}
"""
router_prompt = PromptTemplate.from_template(router_prompt_template)
router_chain = router_prompt | llm | StrOutputParser()

def router_node(state: AgentState):
    """This is the Router node. It decides the next step."""
    print("--- Executing Router Node ---")
    question = state['question']
    route_decision = router_chain.invoke({"question": question})
    cleaned_decision = route_decision.strip().lower()
    print(f"Router Decision: '{cleaned_decision}'")
    return {"route": cleaned_decision}

# --- 6. (NEW) Define the Synthesizer Node ---

synthesizer_prompt_template = """
You are an expert answer synthesizer. Your job is to take a user's question
and the raw data gathered by an agent, and write a clean, helpful, final answer.

The user asked:
{question}

The agent found this raw data:
{agent_data}

Based on the raw data, provide a clear and concise answer.
If the agent data is an error message, just say "I'm sorry, I couldn't find an answer for that."

Also, if possible, try to extract any key relationships from the text in the format (Subject)-[Relationship]->(Object). For example: (Stakeholders)-[influence]->(Projects). List these at the end under a 'Relationships:' heading.
"""

synthesizer_prompt = PromptTemplate.from_template(synthesizer_prompt_template)
synthesizer_chain = synthesizer_prompt | llm | StrOutputParser()

def synthesizer_node(state: AgentState):
    """
    This is the Synthesizer node. It takes the agent's output
    and generates a clean, final answer.
    """
    print("--- Executing Synthesizer Node ---")
    
    # Get the data from the state
    question = state['question']
    # Check which agent ran by seeing which answer key is filled
    if state.get('doc_answer'):
        agent_data = state['doc_answer']
    elif state.get('web_answer'):
        agent_data = state['web_answer']
    else:
        agent_data = "No data found."
    
    # Run the synthesizer chain
    final_answer = synthesizer_chain.invoke({
        "question": question,
        "agent_data": agent_data
    })
    
    print(f"Final Answer: {final_answer}")
    return {"final_answer": final_answer}


# --- 7. Define the Conditional Edge Logic (No changes) ---
def decide_route(state: AgentState):
    """This function is the "conditional edge." It reads the 'route'
    from the state and returns the name of the next node to run."""
    route = state.get("route", "")
    if "document_search" in route:
        return "DocumentAgent"
    elif "web_search" in route:
        return "WebSearchAgent"
    else:
        print("Fallback: Defaulting to WebSearchAgent")
        return "WebSearchAgent"

# --- 8. (NEW) Build the Graph with the Synthesizer ---

workflow = StateGraph(AgentState)

# Add all four nodes
workflow.add_node("Router", router_node)
workflow.add_node("DocumentAgent", document_agent_node)
workflow.add_node("WebSearchAgent", web_search_node)
workflow.add_node("Synthesizer", synthesizer_node) # (NEW)

# Set the entry point
workflow.set_entry_point("Router")

# Add the conditional edges
workflow.add_conditional_edges(
    "Router",
    decide_route,
    {
        "DocumentAgent": "DocumentAgent",
        "WebSearchAgent": "WebSearchAgent"
    }
)

# (NEW) Re-wire the graph:
# Both agents now go to the Synthesizer
workflow.add_edge("DocumentAgent", "Synthesizer")
workflow.add_edge("WebSearchAgent", "Synthesizer")

# The Synthesizer goes to the END
workflow.add_edge("Synthesizer", END)

# Compile the graph
app = workflow.compile()

# --- 9. Run it! (One test at a time) ---

# ---
# TEST 1: (Document Question) - Comment this block out
# ---
# print("Running the Agent graph (Test 1: Document)...")
# initial_input = {"question": "What is a project stakeholder?"}


# ---
# TEST 2: (Web Question) - Uncomment this block to run Test 2
# ---
print("Running the Agent graph (Test 2: Virat Kohli runs)...")
initial_input = {"question": "How many runs does Virat Kohli have in ODIs till date?"}


try:
    # ---
    # Run the graph with the selected input
    # ---
    final_state = app.invoke(initial_input)

    print("\n--- Graph Finished ---")
    print(f"Final State: {final_state}")

except Exception as e:
    print("\n--- SCRIPT CRASHED ---")
    print(f"An error occurred: {e}")
    # Import traceback to print the full error details
    import traceback
    traceback.print_exc()