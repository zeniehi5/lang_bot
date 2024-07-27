from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ResponseGenerator:
    def __init__(self):
        prompt = PromptTemplate.from_template(
            """
                You are an assistant for question-answering tasks. 
                This application is a chatbot for the DFOCUS Solution Team's AI study. 
                You can provide information on company regulations or internal contact networks, and you are particularly confident in answering questions about the Constitution.
                Use the following pieces of retrieved context and chat history to answer the question.
                If you don't know the answer, just say that you don't know. 
                Answer in Korean.

                #Question: 
                {question} 
                #Context: 
                {context}

                #Chat History:
                {chat_history}

                #Answer:
            """
        )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            streaming=True
        )
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        self.chat_history = []

    def generate(self, question, context):
        chat_history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.chat_history])
        input = {
            "context": context,
            "question": question,
            "chat_history": chat_history_str
        }       
        print("question : ", question)
        print("context : ", context)
        print("chat_history : ", chat_history_str)
        response = self.chain.invoke(input)

        self.chat_history.append((question, response))

        if len(self.chat_history) > 5:
            self.chat_history.pop(0)
            
        return response