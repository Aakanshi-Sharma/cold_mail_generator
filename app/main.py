import streamlit as st
from langchain.document_loaders import SeleniumURLLoader
from utils import clean_text
from chain import Chain
from portfolio import Portfolio


def load_dynamic_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    job_details = documents[0].metadata["description"]
    return job_details


def create_streamlit_app(chain, portfolio):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter the URL", placeholder="https://example.com")
    submit_button = st.button("Submit")
    if submit_button:
        with st.spinner('Wait for it...'):
            job_details = load_dynamic_page(url_input)
            job_details = clean_text(job_details)
            portfolio.load_portfolio()
            jobs = chain.extract_job_details(job_details)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_search(skills)
                email = chain.write_email("Default_name", job, links)
                st.code(email, language='markdown')


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio)