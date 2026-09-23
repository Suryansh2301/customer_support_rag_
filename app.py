import streamlit as st
from rag_chain import answer_query, preload_pipeline

st.set_page_config(page_title="Customer Support Assistant", page_icon="💬")


@st.cache_resource(show_spinner="Loading knowledge base and models...")
def _warm_up():
    """Loads embeddings, vector store, reranker, and the LLM client BEFORE
    the chat UI is shown. Streamlit caches this across every rerun and every
    user session in this process, so it truly only happens once per start --
    not per message, and not per user."""
    preload_pipeline()
    return True


_warm_up()  

st.title("💬 Customer Support Assistant")
st.caption(
    "Built-in knowledge base -- no file upload needed. "
    "Ask about orders, refunds, shipping, payments, and more."
)

with st.sidebar:
    st.markdown("### Observability")
    st.caption("Every question here is logged with latency + matched intents.")
    st.code("streamlit run observability/dashboard.py", language="bash")
    st.caption("Run that in a separate terminal to see the live metrics dashboard.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your support assistant. How can I help with your order today?",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking the knowledge base..."):
            history = st.session_state.messages[:-1]
            result = answer_query(user_input, history)  

            answer = result["answer"]
            sources = result["sources"]
            
            timings = result.get("timings", {})

        st.markdown(answer)
        if sources:
            with st.expander("Matched knowledge base entries"):
                for s in sources:
                    st.markdown(f"- **[{s['intent']}]** {s['question']}")
        if timings:
            st.caption(f"⏱ {timings.get('total', 0)}s total")

    st.session_state.messages.append({"role": "assistant", "content": answer})
