from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Changed from OpenAI to ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.callbacks import get_openai_callback
import time

# Ensure you set the OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

def create_custom_prompt():

    
    system_template = """You are an AI expert with deep expertise in CRS (Cytokine Release Syndrome), CRS drug discovery, CRS biomarkers, and cutting-edge CRS research.

Instructions:
1. Carefully analyze the provided context and synthesize a comprehensive response
2. Draw upon the specific details from the retrieved documents
3. Do not mention the citation numbers.
3. Present information in a clear, organized manner"""

    human_template = """Context from retrieved documents:
{context}

User Question: {question}

Please provide a detailed response that:
- Directly addresses the user's question
- Incorporates specific examples and evidence from the context
- Organizes information in a logical structure"""

    # Create chat prompt with system and human messages
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

    return chat_prompt

def load_faiss_index(faiss_index_file: str):
    return FAISS.load_local(faiss_index_file, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def create_retrieval_qa_chain(faiss_index):
    # Initialize components
    retriever = faiss_index.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    
    llm = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-4o-mini",  # Using GPT-4 for better synthesis
        max_tokens=4000,
        streaming=True
    )
    
    prompt = create_custom_prompt()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True
        },
        return_source_documents=True  
    )
    
    return qa_chain

def main():
    faiss_index_file = "vector_store_faiss"
    faiss_index = load_faiss_index(faiss_index_file)

    # Create QA chain
    qa_chain = create_retrieval_qa_chain(faiss_index)

    print("Welcome to the CRS Expert Q&A System. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question about CRS: ")
        # if query.lower() == 'exit':
        #     print("Goodbye!")
        #     break
        try:
            start_time = time.time()
            with get_openai_callback() as cb:
                response = qa_chain({"query": query})
                end_time = time.time()
                elapsed_time = end_time - start_time

                print("\nResponse:", response['result'])

                print(f"\nTiming Analysis:")
                print(f"- Response Generation Time: {elapsed_time:.2f} seconds")

                print("\n=== Token Usage Analysis ===")
                print("\nContext Window Usage:")
                print(f"- Prompt Tokens (Input): {cb.prompt_tokens}")
                print(f"- Completion Tokens (Output): {cb.completion_tokens}")
                print(f"- Total Tokens Used: {cb.total_tokens}")
                print(f"- Context Window Available: {128000 - cb.total_tokens} tokens remaining")
                
                print("\nResponse Length Analysis:")
                print(f"- Response Tokens Used: {cb.completion_tokens}")
                print(f"- Max Allowed Response: 4,096 tokens")
                print(f"- Response Token Headroom: {4096 - cb.completion_tokens} tokens")
                

                context_usage_percent = (cb.total_tokens / 128000) * 100
                response_usage_percent = (cb.completion_tokens / 4096) * 100
                
                print(f"\nUsage Percentages:")
                print(f"- Context Window: {context_usage_percent:.1f}% used")
                print(f"- Response Length: {response_usage_percent:.1f}% of maximum")
                
                print(f"\nCost: ${cb.total_cost:.4f}")
                

                if context_usage_percent > 80:
                    print("\n⚠️ Warning: Context window usage is high!")
                if response_usage_percent > 80:
                    print("\n⚠️ Warning: Response length is approaching maximum!")



            #response = qa_chain({"query": query})
            
            #print("\nExpert Response:", response['result'])
            
            # Optionally print sources
            # if 'source_documents' in response:
            #     print("\nSources consulted:")
            #     for idx, doc in enumerate(response['source_documents'], 1):
            #         print(f"\nSource {idx}:")
            #         print(doc.page_content[:200] + "...")
                    
        except Exception as e:
            print("\nAn error occurred:", e)

if __name__ == "__main__":
    main()