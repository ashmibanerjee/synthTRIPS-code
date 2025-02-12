import base64
import json
import os
from typing import Optional

import vertexai
from dotenv import load_dotenv
from google.auth import default, transport
from google.oauth2 import service_account
from vertexai import generative_models

load_dotenv()
if "VERTEXAI_PROJECTID" in os.environ:
    VERTEXAI_PROJECT = os.environ["VERTEXAI_PROJECTID"]


def decode_service_key():
    encoded_key = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    original_service_key = json.loads(base64.b64decode(encoded_key).decode('utf-8'))
    if original_service_key:
        return original_service_key
    return None


def initialize_vertexai_params(location: Optional[str] = "us-central1"):
    creds_file_name = os.getcwd() + "/.config/application_default_credentials.json"
    if not os.path.exists(os.path.dirname(creds_file_name)):
        credentials = decode_service_key()
        with open(creds_file_name, 'w', encoding='utf-8') as file:
            json.dump(credentials, file, ensure_ascii=False, indent=4)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file_name

    service_account.Credentials.from_service_account_file(
        filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    vertexai.init(project=VERTEXAI_PROJECT, location=location)


def get_default_config() -> tuple[dict, dict]:
    default_gen_config = {
        "temperature": 0.49,
        "max_output_tokens": 1024,
    }
    default_safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }
    return default_gen_config, default_safety_settings


def generate_acess_tokens():
    credentials, _ = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    auth_request = transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials
