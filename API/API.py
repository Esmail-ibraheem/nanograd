from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import subprocess
import random
import base64
import io

from nanograd.models.stable_diffusion import model_loader, pipeline
from nanograd.models.GPT.tokenizer import tokenize

from fastapi.middleware.cors import CORSMiddleware

# Import for authentication
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Configure devices
DEVICE = "cuda"
ALLOW_CUDA = False
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Load Stable Diffusion model
tokenizer_vocab_path = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\\data\\tokenizer_vocab.json")
tokenizer_merges_path = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\\data\\tokenizer_merges.txt")
model_file = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\\data\\v1-5-pruned-emaonly.ckpt")

tokenizer = CLIPTokenizer(str(tokenizer_vocab_path), merges_file=str(tokenizer_merges_path))
models = model_loader.preload_models_from_standard_weights(str(model_file), DEVICE)

# Blueprints (assuming they are defined somewhere in your code)
blueprints = {
    "Visual Story": {
        "sd_prompts": [
            "A futuristic city skyline at dusk, flying cars, neon lights, cyberpunk style",
            "A bustling marketplace in a futuristic city, holograms, diverse crowd",
            "A serene park in a futuristic city with advanced technology blending with nature"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe a futuristic city that blends natural elements with advanced technology.",
            "Write about an advanced cityscape with unique technological elements.",
            "Imagine a futuristic metropolis where nature and technology harmoniously coexist."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    # ... (other blueprints)
}

# JWT configuration
SECRET_KEY = "your-secret-key"  # Replace with a strong secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer instance
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dummy user database (replace with a real database in production)
fake_users_db = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderland",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash("password123"),
        "disabled": True,
    },
    "bob": {
        "username": "bob",
        "full_name": "Bob Builder",
        "email": "bob@example.com",
        "hashed_password": pwd_context.hash("mypassword"),
        "disabled": False,
    },
}

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None

class UserInDB(User):
    hashed_password: str

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Default to 15 minutes if no expire time provided
        expire = datetime.utcnow() + timedelta(seconds=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Login endpoint
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Access token expires in ACCESS_TOKEN_EXPIRE_MINUTES
    access_token_expires = timedelta(seconds=ACCESS_TOKEN_EXPIRE_MINUTES)
    # Data to encode in the token
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Dependency to get the current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    # Get the user from the database
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return user

# Dependency to get the current active user
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Models for your endpoints
class ImageGenerationRequest(BaseModel):
    prompt: str
    cfg_scale: float
    num_inference_steps: int
    sampler: str

class TextGenerationRequest(BaseModel):
    model_name: str
    prompt: str

class BlueprintRequest(BaseModel):
    blueprint_name: str

class TokenizeRequest(BaseModel):
    text: str

class ChatbotRequest(BaseModel):
    question: str

# Protected endpoints
@app.post("/generate_image")
async def generate_image(request: ImageGenerationRequest, current_user: User = Depends(get_current_active_user)):
    uncond_prompt = ""
    do_cfg = True
    input_image = None
    strength = 0.9
    seed = 42

    output_image = pipeline.generate(
        prompt=request.prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=request.cfg_scale,
        sampler_name=request.sampler,
        n_inference_steps=request.num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image = Image.fromarray(output_image)

    # Convert PIL Image to base64 string
    buffered = io.BytesIO()
    output_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image": img_str}

@app.post("/chat_with_ollama")
async def chat_with_ollama(request: TextGenerationRequest, current_user: User = Depends(get_current_active_user)):
    command = ['ollama', 'run', request.model_name, request.prompt]
    result = subprocess.run(command, capture_output=True, text=True)
    return {"response": result.stdout}

@app.post("/apply_blueprint")
async def apply_blueprint(request: BlueprintRequest, current_user: User = Depends(get_current_active_user)):
    if request.blueprint_name in blueprints:
        bp = blueprints[request.blueprint_name]
        sd_prompts = random.choice(bp["sd_prompts"])
        sd_cfg_scale = random.choice(bp["sd_cfg_scales"])
        sd_num_inference_steps = random.choice(bp["sd_num_inference_steps"])
        sd_sampler = random.choice(bp["sd_samplers"])
        ollama_prompts = random.choice(bp["ollama_prompts"])
        ollama_model = random.choice(bp["ollama_models"])
        return {
            "sd_prompt": sd_prompts,
            "sd_cfg_scale": sd_cfg_scale,
            "sd_num_inference_steps": sd_num_inference_steps,
            "sd_sampler": sd_sampler,
            "ollama_model": ollama_model,
            "ollama_prompt": ollama_prompts
        }
    raise HTTPException(status_code=404, detail="Blueprint not found")

@app.post("/tokenize")
async def tokenize_text(request: TokenizeRequest, current_user: User = Depends(get_current_active_user)):
    tokens = tokenize(request.text)
    return {"tokens": tokens}

@app.post("/chatbot_arabic")
async def chatbot_arabic(request: ChatbotRequest, current_user: User = Depends(get_current_active_user)):
    from nanograd.models import ollama
    response = ollama.run(request.question)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
