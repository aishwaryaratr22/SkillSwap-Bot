# SkillSwap-Bot


🎯 SkillSwap Marketplace – RAG-Powered Chatbot


📌 Problem Statement

In today's fast-paced world, many people want to learn new skills or share what they know — but finding the right match (i.e., someone who teaches what you want to learn, or wants to learn what you teach) can be frustrating, especially on general-purpose platforms.

This project aims to build a SkillSwap platform where:
Learners can find the right teachers.
Teachers can find interested learners.
The matching is done semantically, not just by keyword search.


🚧 Current Progress Status

✅ Core backend logic for storing and embedding user-submitted skills (using SQLite + sentence-transformers).
✅ FAISS integration for semantic retrieval based on user queries.\\
✅ LangChain + LLaMA3 via Ollama to generate natural language responses based on retrieved matches (RAG).
✅ Streamlit frontend to add skills and interact with the bot.
✅ Basic UI with interactive elements and match display.



💡 How the Prototype Solves the Problem

👉🏻 Users submit skills (they can either teach or want to learn a skill).
👉🏻 Each skill description is converted to an embedding vector using a sentence transformer.
👉🏻 When a user asks for help or wants to offer help:
👉🏻 The system uses FAISS to find semantically similar matches with the opposite role (i.e., learners see teachers, teachers see learners).=
👉🏻 The top matches are passed to an LLM (LLaMA3) using a custom prompt (RAG: Retrieval-Augmented Generation).
👉🏻 The chatbot then generates a friendly, natural-language response suggesting relevant matches.


🧰 Technologies & Tools Used

Tool/Library	                                                                        Purpose

Streamlit	                                                        Web app interface
SQLite3	                                                          Lightweight database for storing skill records
Sentence-Transformers (all-MiniLM-L6-v2)	                        Converts skill descriptions and queries into embeddings
FAISS	                                                            Finds nearest skill matches using embedding similarity
LangChain	                                                        Framework to integrate prompts and LLMs
Ollama + LLaMA3	                                                  Local language model used to generate chatbot responses
Python	                                                          Programming language for full-stack logic

