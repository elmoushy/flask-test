import json
from flask import Flask, request, jsonify

# LangChain and related imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# For GPT4All Embeddings
from langchain_community.embeddings import GPT4AllEmbeddings

# Import the Groq LLM
from langchain_groq import ChatGroq

###############################################################################
# 1) بيانات التهيئة
###############################################################################
JSON_FILE_PATH = "data.json"

# بتحميل البيانات من ملف JSON
def load_dataset(json_file_path: str) -> dict:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# تحويل البيانات إلى مستندات
def build_documents(data: dict):
    documents = []

    # معلومات الوحدة
    module_text = f"اسم الوحدة: {data.get('module', '')}\n" \
                  f"الوصف: {data.get('description', '')}"
    documents.append(Document(page_content=module_text, metadata={"type": "module_info"}))

    # السيناريوهات
    for scenario in data.get("scenarios", []):
        scenario_text = f"سيناريو: {scenario.get('scenario', '')}\n"
        for step in scenario.get("steps", []):
            scenario_text += f"- {step}\n"

        for item in scenario.get("expected_bot_responses", []):
            scenario_text += f"\nسؤال المستخدم: {item.get('user_input', '')}\n"
            scenario_text += f"رد البوت: {item.get('bot_response', '')}\n"

        documents.append(Document(page_content=scenario_text, metadata={"type": "scenario"}))

    return documents

# إنشاء المتجر المتجهي (ChromaDB) باستخدام GPT4All
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = [
        Document(page_content=chunk, metadata=doc.metadata)
        for doc in documents
        for chunk in text_splitter.split_text(doc.page_content)
    ]

    embedding_model = GPT4AllEmbeddings()
    vector_store = Chroma.from_documents(splitted_docs, embedding_model)
    return vector_store

# استرجاع السياق
def retrieve_context(query, vector_store, k=3):
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

# تحميل نموذج Groq's llama-3.3-70b-versatile
def load_groq_model():
    """
    Loads the 'llama-3.3-70b-versatile' model from Groq
    with the API key passed in directly.
    """
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.4,
        api_key="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"
    )
    return llm

# يستخدم لاسترجاع السياق وتوليد الإجابة
def rag_answer(query, vector_store, groq_llm, k=3):
    # 1) استرجاع السياق
    context_list = retrieve_context(query, vector_store, k)
    context_str = "\n".join(context_list)

    # 2) بناء الرسائل بما في ذلك السياق وسؤال المستخدم
    messages = [
        SystemMessage(content=f"السياق:\n{context_str}\n\n"),
        HumanMessage(content=f"السؤال: {query}\n")
    ]

    # 3) توليد الإجابة باستخدام النموذج
    answer = groq_llm(messages)
    return answer.content

###############################################################################
# 2) تهيئة Flask
###############################################################################
app = Flask(__name__)

# تحميل البيانات وإنشاء المتجر المتجهي وتحميل النموذج لمرة واحدة
data = load_dataset(JSON_FILE_PATH)
docs = build_documents(data)
vector_store = create_vector_store(docs)
groq_llm = load_groq_model()

###############################################################################
# 3) تعريف Endpoint لاستقبال السؤال وإعادة الإجابة
###############################################################################
@app.route('/ask', methods=['POST'])
def ask_model():
    """
    Expects a JSON payload like: { "question": "اكتب سؤالك هنا" }
    Returns a JSON response like: { "answer": "نص الإجابة" }
    """
    try:
        incoming_data = request.get_json(force=True)
        user_question = incoming_data.get("question", "")

        if not user_question:
            return jsonify({"error": "No question provided."}), 400

        # استدعاء الدالة RAG للإجابة
        answer = rag_answer(user_question, vector_store, groq_llm, k=3)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

###############################################################################
# 4) تشغيل السيرفر
###############################################################################

if __name__ == '__main__':
    app.run()
    #app.run(host='0.0.0.0', port=5000, debug=False)