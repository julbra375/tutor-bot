from openai import OpenAI

class OpenAIGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, context: str) -> str:
        prompt = f"""Context from a physics textbook: {context}
        Question: {question}. Be specific, sticking closely to the context. 
        Include equations where relevant. Make sure to cite source and page number where relevant. 
        Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

