import csv
import openai
import os

class OpenAIModel:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai = openai
        
    def convert_to_vector(self, text):
        response = self.openai.Embedding.create(
          model="text-embedding-ada-002",
          input=text
        )
        
        # 消費したトークン数をCSVに記録
        filename = 'output/openai_usage_tokens.csv'
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Text', 'Total Tokens'])
            writer.writerow([text, response['usage']['total_tokens']])
        
        # vectorを返却
        return response['data'][0]['embedding']
    
