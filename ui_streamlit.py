# ui_streamlit.py
import streamlit as st
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="RAG Demo - IPCC AR6",
    page_icon="🌍",
    layout="wide"
)

# Titre principal
st.title("🌍 RAG Demo — IPCC AR6")
st.subheader("Retrieval-Augmented Generation with Ollama + LangChain")

st.markdown("""
This application uses local LLMs (via Ollama) to answer questions about IPCC AR6 climate reports.
Ask any question about climate science, and the system will retrieve relevant information from the documents.
""")

st.divider()

# URL de l'API backend
API_URL = "http://localhost:8000/ask"

# Zone de saisie de la question
question = st.text_input(
    "❓ Ask a question about the IPCC reports:",
    placeholder="e.g., What are the main causes of climate change?",
    key="question_input"
)

# Bouton pour soumettre la question
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

# Gestion du bouton Clear
if clear_button:
    st.rerun()

# Traitement de la question
if ask_button and question:
    with st.spinner("🤔 Thinking... Retrieving relevant information and generating answer..."):
        try:
            # Appel à l'API
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=60
            )
            
            if response.ok:
                data = response.json()
                
                # Affichage de la réponse
                st.success("✅ Answer generated successfully!")
                
                st.markdown("### 💡 Answer")
                st.markdown(f"**{data['answer']}**")
                
                st.divider()
                
                # Affichage des sources
                st.markdown("### 📚 Sources")
                sources = data.get("sources", [])
                
                if sources:
                    for idx, source in enumerate(sources, 1):
                        with st.expander(f"📄 Source {idx}"):
                            st.write(f"**File:** {source.get('source', 'Unknown')}")
                            st.write(f"**Page:** {source.get('page', 'N/A')}")
                else:
                    st.info("No sources returned.")
                    
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write(f"Details: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Error: Cannot connect to the backend API.")
            st.info("Make sure the FastAPI server is running on http://localhost:8000")
            st.code("uvicorn app:app --reload --port 8000", language="bash")
            
        except requests.exceptions.Timeout:
            st.error("❌ Timeout Error: The request took too long.")
            st.info("The model might be processing a complex query. Try again or simplify your question.")
            
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

elif ask_button and not question:
    st.warning("⚠️ Please enter a question before clicking 'Ask'.")

# Sidebar avec informations
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG application demonstrates:
    - **Local LLM** inference with Ollama
    - **Vector search** using ChromaDB
    - **Document retrieval** from IPCC AR6 reports
    - **FastAPI** backend
    - **Streamlit** frontend
    """)
    
    st.divider()
    
    st.header("📖 How to Use")
    st.markdown("""
    1. Enter your question in the text box
    2. Click **Ask** to get an answer
    3. View the answer and sources
    4. Use **Clear** to start a new query
    """)
    
    st.divider()
    
    st.header("🔧 System Status")
    
    # Test de connexion à l'API
    try:
        test_response = requests.get("http://localhost:8000/docs", timeout=2)
        if test_response.ok:
            st.success("✅ Backend API is running")
        else:
            st.error("❌ Backend API error")
    except:
        st.error("❌ Backend API is not running")
        st.code("uvicorn app:app --reload --port 8000", language="bash")
    
    st.divider()
    
    st.header("📚 Dataset")
    st.markdown("""
    **IPCC AR6 Reports:**
    - WGI Summary for Policymakers
    - AR6 Synthesis Report (Full Volume)
    - AR6 SYR Summary for Policymakers
    """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Built with ❤️ using Ollama, LangChain, FastAPI, and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)