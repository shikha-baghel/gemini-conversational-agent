import os
import google.generativeai as genai
from typing import List, Dict

if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, value = line.strip().split("=", 1)
                os.environ[key.strip()] = value.strip()

class GeminiChatbot:
    def __init__(self, api_key: str):
        """
        Initialize the Gemini
        :param api_key: Google Gemini API key
        """
        if not api_key:
            raise ValueError("Error: Unable to find API key. Please check your .env file format!")
            
        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 512,
        }
        
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        self.conversation_history: List[Dict[str, str]] = []         

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response using Gemini API
        
        :param user_input: User's message
        :return: AI's response
        """
        try:
            chat_history = [
                {"role": msg.get("role", "user"), "parts": [msg.get("parts", msg.get("text", ""))]} 
                for msg in self.conversation_history
            ]
            
            chat = self.model.start_chat(history=chat_history)  
            response = chat.send_message(user_input)  
            
            self.conversation_history.append({
                "role": "user",
                "parts": user_input
            })
            self.conversation_history.append({
                "role": "model",
                "parts": response.text
            })
            
            return response.text
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, I'm having trouble processing your request."
    
    def chat(self):
        """
        chat loop WHILE :-
        """
        print("My_AI: Hello! I'm ready to chat.")
        
        while True:
            try:
                user_input = input("You: ")
                
                if user_input.lower() in ['bye', 'goodbye', 'exit']:   
                    print("My_AI: Goodbye!")
                    break
                
                response = self.generate_response(user_input)
                print("My_AI:", response)
                
            except (KeyboardInterrupt, EOFError):
                print("\nMy_AI: Goodbye!")
                break

def main():
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "GEMINI_API_KEY" in line and "=" in line:
                    api_key = line.strip().split("=", 1).strip()

    chatbot = GeminiChatbot(api_key) 
    chatbot.chat()

if __name__ == "__main__":
    main()