import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, model="gpt-4", system_prompt="Jsi pomocná jednotka systému EVO."):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = {"role": "system", "content": system_prompt}
        self.history = []

    def ask(self, user_input, functions=None):
        messages = [self.system_prompt] + self.history + [{"role": "user", "content": user_input}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=functions or None,
                function_call="auto" if functions else None,
                temperature=0.7
            )
            message = response.choices[0].message
            self.history.append({"role": "user", "content": user_input})
            self.history.append(message.dict())
            return message.content or "Žádná odpověď"
        except Exception as e:
            return f"Chyba OpenAI: {e}"

    def reset_history(self):
        self.history = []