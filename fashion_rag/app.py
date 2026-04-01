"""
Streamlit chat UI.
Start: streamlit run app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Fashion Assistant", page_icon="👗", layout="wide")
st.title("👗 Fashion Product Assistant")

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    gender = st.selectbox(
        "Gender", ["Any", "Men", "Women", "Boys", "Girls", "Unisex"]
    )
    category = st.selectbox(
        "Category", ["Any", "Apparel", "Footwear", "Accessories", "Personal Care"]
    )
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about products, colours, styles…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching products…"):
            try:
                resp = requests.post(
                    API_URL,
                    json={
                        "question": prompt,
                        "gender"  : None if gender == "Any" else gender,
                        "category": None if category == "Any" else category,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data   = resp.json()
                answer = data["answer"]
                st.write(answer)

                col1, col2 = st.columns(2)
                col1.caption(f"⏱ {data['latency_ms']} ms")
                col2.caption("⚡ cached" if data["cached"] else "")

                with st.expander("Source products"):
                    for src in data["sources"]:
                        st.json(src)

            except requests.exceptions.ConnectionError:
                answer = "Could not reach the API. Is `uvicorn api:app` running?"
                st.error(answer)
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
