from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel
from openai import OpenAI
import config
import json
import os

# create server
app = FastAPI(title="API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create client
client = OpenAI(base_url=config.BASE_URL, api_key=os.getenv("HF_TOKEN"))
model = config.MODEL["qwen"]


# interface for request
class TextRequest(BaseModel):
    text: str


@app.post("/summary")
def summary(text: TextRequest):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Summarize the following text: {text.text}",
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
    )
    return json.loads(response.choices[0].message.content)


@app.post("/translate")
def translate(text: TextRequest):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Translate the following text to Tamil: {text.text}",
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
    )
    return json.loads(response.choices[0].message.content)


@app.post("/keywords")
def keywords(text: TextRequest):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Extract keywords from the following text: {text.text}",
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
    )
    return json.loads(response.choices[0].message.content)

