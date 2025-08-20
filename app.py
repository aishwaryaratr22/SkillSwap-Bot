import streamlit as st
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DB_NAME = "skills.db"

llm = Ollama(model="llama3")

template = """
You are a helpful assistant in a SkillSwap marketplace.
Based on the user's query and the matching skill profiles, generate a friendly and useful response.

User Query:
{query}

Matched Skills:
{matches}

Respond in a conversational tone.
"""

prompt = PromptTemplate(template=template, input_variables=["query", "matches"])
chain = LLMChain(llm=llm, prompt=prompt)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        create table if not exists skills (
        id integer primary key autoincrement,
        name text,
        description text,
        role text,
        embedding blob
        )
    """)
    conn.commit()
    conn.close()

def add_skill(name, description, role):
    emb = EMBED_MODEL.encode(description).astype(np.float32)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO skills (name, description, role, embedding) VALUES (?, ?, ?, ?)",
              (name, description, role, emb.tobytes()))
    conn.commit()
    conn.close()

def fetch_skills():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, description, role, embedding FROM skills")
    data = c.fetchall()
    conn.close()
    return data

def retrieve_matches(query, role_filter, k=3):
    skills = fetch_skills()
    if not skills:
        return []

    embs = []
    meta = []

    for name, desc, role, emb_blob in skills:
        if role != role_filter:
            continue
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        embs.append(emb)
        meta.append((name, desc))

    if not embs:
        return []

    dim = len(embs[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs))

    q_emb = EMBED_MODEL.encode([query]).astype(np.float32)
    distances, indices = index.search(q_emb, k)

    results = []
    for i in indices[0]:
        if i < len(meta):
            results.append({
                "name": meta[i][0],
                "description": meta[i][1]
            })
    return results

def main():
    st.set_page_config("SkillSwap RAG Bot", layout="centered")
    st.title("🎯 SkillSwap Marketplace")

    init_db() 

    menu = st.sidebar.radio("Navigation", ["➕ Add Skill", "💬 Talk to Bot"])

    if menu == "➕ Add Skill":
        st.subheader("Add Your Skill or Need")
        name = st.text_input("Your Name")
        description = st.text_area("Describe the skill you want to share or learn")
        role = st.selectbox("Are you offering or seeking?", ["teacher", "learner"])

        if st.button("Add Skill"):
            if name and description:
                add_skill(name, description, role)
                st.success("✅ Skill added successfully!")
            else:
                st.warning("Fill all fields!")

    elif menu == "💬 Talk to Bot":
        st.subheader("SkillSwap Bot")

        user_query = st.text_input("What skill are you looking for or offering?")
        user_role = st.selectbox("You are a...", ["learner", "teacher"])

        if st.button("Find Skills"):
            if not user_query:
                st.warning("Please enter a query.")
            else:
                opposite = "teacher" if user_role == "learner" else "learner"
                matches = retrieve_matches(user_query, opposite)

                if matches:
                    match_text = "\n".join([f"{m['name']}: {m['description']}" for m in matches])
                    response = chain.run(query=user_query, matches=match_text)
                    st.markdown("### 🤖 Bot Response")
                    st.success(response)

                    with st.expander("🔍 Matching Profiles"):
                        for m in matches:
                            st.markdown(f"- **{m['name']}**: {m['description']}")
                else:
                    st.info("No matches found. Try a different query or add more skills.")

if __name__ == "__main__":
    main()