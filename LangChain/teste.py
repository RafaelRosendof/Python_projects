API_KEY = "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"

from langchain import HuggingFaceHub

llm = HuggingFaceHub(repo_id = "microsoft/Phi-3.5-mini-instruct", huggingfacehub_api_token = API_KEY)

llm_response = llm.generate(['Tell me a joke about data scientist',

'Tell me a joke about recruiter',

'Tell me a joke about psychologist'])

print(llm_response[1])