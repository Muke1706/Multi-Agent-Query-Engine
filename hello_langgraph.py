from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# --- 1. Define the State ---
# This is the "whiteboard" that all nodes share.
# It's a dictionary that must contain a "number" key.
class GraphState(TypedDict):
    number: int

# --- 2. Define the Nodes ---
# Nodes are just Python functions. They take the current state
# and return a dictionary with the keys they want to update.

def node_a(state: GraphState):
    """ Adds 1 to the number. """
    print("--- Executing Node A ---")
    current_number = state['number']
    current_number += 1
    
    # Return the updated state
    return {"number": current_number}

def node_b(state: GraphState):
    """ Multiplies the number by 2. """
    print("--- Executing Node B ---")
    current_number = state['number']
    current_number *= 2
    
    # Return the updated state
    return {"number": current_number}

# --- 3. Build the Graph ---

# This is where we wire everything together.
workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("A", node_a)
workflow.add_node("B", node_b)

# Add the edges (the "arrows")
workflow.set_entry_point("A")  # The graph will start at Node A
workflow.add_edge("A", "B")      # After Node A, go to Node B
workflow.add_edge("B", END)      # After Node B, stop (END)

# --- 4. Compile the Graph ---
# This turns our blueprint into a runnable application.
app = workflow.compile()

# --- 5. Run it! ---
print("Running Langgraph 'Hello, World'...")

# 'invoke' runs the graph from start to finish.
# We pass in the initial state.
initial_input = {"number": 10}
final_state = app.invoke(initial_input)

print("\n--- Graph Finished ---")
print(f"Initial input: {initial_input}")
print(f"Final state: {final_state}")

# --- Let's try another one ---
print("\nRunning with a different input...")
initial_input_2 = {"number": 3}
final_state_2 = app.invoke(initial_input_2)

print("\n--- Graph Finished ---")
print(f"Initial input: {initial_input_2}")
print(f"Final state: {final_state_2}")