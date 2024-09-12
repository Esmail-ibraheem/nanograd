import unittest
from fastapi.testclient import TestClient
from i import app  # Replace 'your_app' with the actual name of your FastAPI application file

class TestYourApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_generate_image(self):
        response = self.client.post("/generate_image", json={
            "prompt": "A futuristic city skyline at dusk, flying cars, neon lights, cyberpunk style",
            "cfg_scale": 9.0,
            "num_inference_steps": 60,
            "sampler": "ddpm"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("image", response.json())

    def test_chat_with_ollama(self):
        response = self.client.post("/chat_with_ollama", json={
            "model_name": "llama3",
            "prompt": "Describe a futuristic city that blends natural elements with advanced technology."
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())

    def test_apply_blueprint(self):
        response = self.client.post("/apply_blueprint", json={
            "blueprint_name": "Visual Story"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("sd_prompt", response.json())
        self.assertIn("ollama_model", response.json())

    def test_tokenize_text(self):
        response = self.client.post("/tokenize", json={
            "text": "Hello, world!"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("tokens", response.json())

    def test_chatbot_arabic(self):
        response = self.client.post("/chatbot_arabic", json={
            "question": "What is the capital of Egypt?"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())

if __name__ == "__main__":
    unittest.main()
