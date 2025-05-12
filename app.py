import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


#create streamlit app
#set up page configuration
st.set_page_config(page_title = "LangChain: Content Summarizer from YouTube or Website")
st.title("Summarize Text from YouTube or Website")
st.subheader("Summarize URL")


#Get the Groq API Key and url (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value = "", type = "password") #to keep password hidden

generic_url = st.text_input("URL", label_visibility="collapsed") # keep password hidden

#Initialize the Groq model outside the application: Gemma model
llm = ChatGroq(model = "Gemma2-9b-It", groq_api_key=groq_api_key)

#prompt template
prompt_template = """
Provide a summary of the following content in about 300 words.
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YouTube or Website!"):
    ##validate the inputs: api key, url, etc.
    if not groq_api_key.strip() or not generic_url.strip(): #strip() removes empty characters
        st.error("Please provide missing information. to get started")
    elif not validators.url(generic_url):
        st.error("This is not a valid URL! Please enter a valid URL, it may be a YouTube URL or website URL")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")
