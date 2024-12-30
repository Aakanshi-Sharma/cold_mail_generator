from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

with open("key.txt", "r") as f:
    api_key = f.readline().strip()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.1-70b-versatile")

    def extract_job_details(self, cleaned_text):
        prompts_extract = PromptTemplate.from_template(
            """ 
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing 
            following keys : `role`, `experience`, `skills` and `description`.
            Only return the valid json
            ###(NO PREAMBLE):
            """
        )

        chain_extract = prompts_extract | self.llm
        res = chain_extract.invoke(input={'page_data': cleaned_text})
        try:
            json_parser = JsonOutputParser()
            json_res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Content is too big.")
        if isinstance(json_res, list):
            return json_res
        else:
            return [json_res]

    def write_email(self, name, job_description, link_list):
        email_prompt = PromptTemplate.from_template("""
                ### JOB DESCRIPTION:
                {job_description}

                ### INSTRUCTION:
                You are {name}, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
                the seamless integration of business processes through automated tools. 
                Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
                process optimization, cost reduction, and heightened overall efficiency. 
                Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
                in fulfilling their needs in proper format.
                Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
                Remember you are {name}. 
                Do not provide a preamble.
                ### EMAIL (NO PREAMBLE):

                """)
        chain_generate = email_prompt | self.llm
        email = chain_generate.invoke({"name": str(name), "job_description": str(job_description), "link_list": link_list})
        return email.content
