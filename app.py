import streamlit as st
# Import the compiled Langgraph app from your existing script
# We assume agent_graph.py and app.py are in the same folder
from agent_graph import app as agent_app

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Engine",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Query Engine")
st.caption("Ask a question about Project Management (from the PDF) or general knowledge/web search.")

# --- Chat Input ---
user_question = st.text_input("Enter your question:")

# --- Agent Invocation & Output ---
if user_question:
    # Display a spinner while the agent is thinking
    with st.spinner("🧠 Thinking... (This might take a moment due to API limits)"):
        try:
            # Prepare the input for the Langgraph app
            initial_input = {"question": user_question}

            # Invoke the Langgraph app (this runs the whole Router -> Agent -> Synthesizer flow)
            final_state = agent_app.invoke(initial_input)

            # Extract the final answer from the state
            final_answer = final_state.get("final_answer", "Sorry, something went wrong.")

            # Display the final answer
            st.markdown("### Answer:")
            st.write(final_answer)

            # (Optional) Display raw agent outputs for debugging/interest
            with st.expander("Show Agent Details"):
                st.write("Route Taken:", final_state.get("route"))
                if final_state.get("doc_answer"):
                    st.write("Document Agent Output:", final_state.get("doc_answer"))
                if final_state.get("web_answer"):
                    st.write("Web Search Agent Output:", final_state.get("web_answer"))

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Optional: Add more detailed error logging if needed
            # import traceback
            # st.exception(traceback.format_exc())