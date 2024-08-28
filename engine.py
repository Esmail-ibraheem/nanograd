from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from nano_engine import generate_image, apply_blueprint

app = FastAPI()

# Mount the 'static' directory to serve JavaScript, CSS, and HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageRequest(BaseModel):
    blueprint_name: str

@app.post("/generate-image/")
async def generate_image_endpoint(request: ImageRequest):
    # Call the function from engine.py to generate an image
    sd_prompts, sd_cfg_scale, sd_num_inference_steps, sd_sampler, ollama_model, ollama_prompt = apply_blueprint(request.blueprint_name)
    image = generate_image(sd_prompts, sd_cfg_scale, sd_num_inference_steps, sd_sampler)
    
    # Save the image and return its path
    image_path = "static/generated_image.png"
    image.save(image_path)
    
    return {"image_url": f"/static/generated_image.png"}

@app.get("/")
async def get_homepage():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
