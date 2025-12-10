# 🌍 RAG Demo — IPCC AR6  
### Retrieval-Augmented Generation with Ollama + LangChain + FastAPI + Streamlit

---

## 📌 Description du Projet

Ce projet implémente une application RAG (Retrieval-Augmented Generation) permettant d’interroger localement les rapports climatiques IPCC AR6 grâce à un modèle LLM exécuté via **Ollama**.  
Le système combine :

- extraction et découpage de PDF  
- embeddings avec un modèle local  
- base vectorielle Chroma  
- pipeline RAG complet  
- backend FastAPI  
- interface utilisateur Streamlit  

L’objectif est d’illustrer un pipeline RAG simple, local et reproductible.

---

## 📁 Structure du Projet

```
project/
│── data/               # PDF IPCC AR6
│── chunks/             # Chunks générés automatiquement
│── vectordb/           # Base vectorielle persistée
│── ingest.py           # Extraction & chunking des PDFs
│── embeddings.py       # Embeddings + stockage vecteur
│── app.py              # Backend FastAPI (RAG)
│── ui_streamlit.py     # Interface Streamlit
│── requirements.txt    # Dépendances Python
│── README.md           # Ce fichier
```

---

## 🧩 Fonctionnalités

- Extraction automatique du texte des PDFs AR6  
- Découpage intelligent en chunks (1000 caractères, overlap 200)  
- Embeddings locaux via *nomic-embed-text*  
- Recherche vectorielle Chroma  
- Pipeline RAG : Retriever + Prompt + LLM (llama3.2:1b)  
- Réponses accompagnées des sources (PDF + page)  
- Interface Streamlit conviviale  

---

## 🛠️ Prérequis

- Python **3.10+**
- **Ollama** installé (https://ollama.com/)
- Modèles Ollama nécessaires :
  ```
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
  ```
- PDF AR6 placés dans le dossier `data/`

---

## 🚀 Installation

### 1️⃣ Cloner le projet
```
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2️⃣ Créer l’environnement Python
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3️⃣ Installer les dépendances
```
pip install -r requirements.txt
```

### 4️⃣ Lancer le daemon Ollama
```
ollama serve &
```

---

## 📥 Étape 1 : Ingestion des PDFs
```
python ingest.py
```

---

## 🧠 Étape 2 : Génération des embeddings
```
python embeddings.py
```

---

## 🧩 Étape 3 : Lancer le backend FastAPI
```
uvicorn app:app --reload --port 8000
```

Accès API :  
➡️ http://localhost:8000/docs

---

## 🎨 Étape 4 : Lancer l’interface Streamlit
```
streamlit run ui_streamlit.py
```

---

## 🧪 Exemples de Questions

- What is the main cause of climate change?  
- What does the AR6 SPM say about sea level rise?  
- Do greenhouse gas emissions need to increase or decrease?

---

## 📚 Rapport inclus

Le projet est accompagné d’un rapport détaillant :

- les choix techniques (chunk size, embeddings, retriever…)  
- les résultats obtenus  
- le fonctionnement global du pipeline RAG  

---

## 🧭 Limitations et Travaux futurs

- Amélioration du prompt  
- Ajout d’un re-ranker  
- Ajout d’un système de feedback utilisateur  
- UI plus avancée en React  
- Comparaison avec LlamaIndex  

---

## 👩‍💻 Auteur
Ton nom ici

---

