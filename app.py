from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

import streamlit as st



chat = ChatGroq(groq_api_key="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",model_name="llama-3.3-70b-versatile")


st.title("Chat with any website")

url_text=st.text_input("Enter a URL")
user_msg=st.text_input("Enter your message")

btn=st.button("Submit")

if btn:

    response= chat.invoke("solve the 0/1 knapsack problem using python")

    docs = WebBaseLoader(url_text)

    extractedText=docs.load()

    textFromFirstWebsite=extractedText[0].page_content

    extract_prompt=PromptTemplate.from_template("""
    ------------------
    The scrapped text is:
    {textFromFirstWebsite}
    -----------------
    Instruction:
    {a}
    """)

    chain=extract_prompt | chat

    res=chain.invoke(input={'textFromFirstWebsite':textFromFirstWebsite,
    "a":user_msg})


    print(res.content)



    st.text(res.content)

    st.markdown(res.content,unsafe_allow_html=True)

