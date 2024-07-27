from flask import Flask, request, jsonify, render_template
from chat.chat_bot import ChatBot
from splitter.text_splitter import TextSplitter
from store.store import Store

app = Flask(__name__)
chatbot = ChatBot()
splitter = TextSplitter()
store = Store()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_documents', methods=['POST'])
def load_documents():
    file = request.files['file']
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)    
    try:
        documents = chatbot.load_documents(file_path)
        chunks = splitter.split_documents(documents)
        store.add_documents(chunks)
        return jsonify({"message": "File loaded successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    question = data['question']
    try:
        response = chatbot.get_response(question, store)
        return jsonify({"response": response}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)