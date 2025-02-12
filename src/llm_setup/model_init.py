import os
from importlib.metadata import PackageNotFoundError

import torch
from anthropic import AnthropicVertex
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import GenerativeModel as GenerativeModelPreview
from vertexai.preview.generative_models import Tool, grounding, GenerationConfig

from src.llm_setup.vertexai_setup import initialize_vertexai_params, generate_acess_tokens

load_dotenv()
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
VERTEXAI_PROJECT = os.getenv("VERTEXAI_PROJECTID")
from typing import Optional


def get_chat_template():
    """
    Returns the chat template format for Hugging Face models.
    """
    return "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- bos_token + '[INST] ' + message[" \
           "'content'].strip() + ' [/INST]' }}{%- elif message['role'] == 'system' %}{{- '<<SYS>>\\n' + message[" \
           "'content'].strip() + '\\n<</SYS>>\\n\\n' }}{%- elif message['role'] == 'assistant' %}{{- '[ASST] ' + " \
           "message['content'] + ' [/ASST]' + eos_token }}{%- endif %}{%- endfor %} "


def initialize_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


class LLMBaseClass:
    """
    Base Class for text generation - user needs to provide the HF model ID while instantiating the class after which
    the generate method can be called to generate responses.
    """

    def __init__(self, model_id, location: Optional[str] = None) -> None:
        self.model_id = model_id
        self.terminators = None
        self.tokenizer = None
        self.temperature = 0.5
        self.tokens = 1024
        self.location = location
        try:
            self.bnb_config = initialize_bnb_config()
        except PackageNotFoundError as e:
            self.bnb_config = None
        self.model = self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the model based on the model_id.
        """
        model_type = self.model_id[0].lower()

        if "gpt-4" in model_type:
            return OpenAI(api_key=OAI_API_KEY)

        elif "claude" in model_type:
            if self.location:
                return AnthropicVertex(region=self.location, project_id=VERTEXAI_PROJECT)
            return AnthropicVertex(region="europe-east5", project_id=VERTEXAI_PROJECT)
        elif model_type in ["gemini-1.5-pro-002", "gemini-2.0-flash-exp"]:
            return self._initialize_vertexai_model()

        elif model_type in ["meta/llama-3.2-90b-vision-instruct-maas"]:
            return self._initialize_vertexai_llama_maas()
        else:  # Assume Hugging Face model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.chat_template = get_chat_template()
            print("tokenizer loaded.")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=self.bnb_config,
            )
            model.generation_config.pad_token_id = tokenizer.pad_token_id

            self.tokenizer = tokenizer
            self.terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            print("updating tokenizer")
            return model

    @staticmethod
    def _initialize_vertexai_llama_maas():
        """
        Initialize Llama model using Vertex AI.
        """
        initialize_vertexai_params(location="us-central1")
        credentials = generate_acess_tokens()
        MODEL_LOCATION = "us-central1"
        MAAS_ENDPOINT = f"{MODEL_LOCATION}-aiplatform.googleapis.com"

        client = OpenAI(
            base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{VERTEXAI_PROJECT}/locations/us-central1/endpoints/openapi",
            api_key=credentials.token,
        )
        return client

    def _initialize_vertexai_model(self):
        """
        Initialize Google Gemini model using Vertex AI.
        """
        if "flash" in self.model_id[0]:
            initialize_vertexai_params(location="us-central1")
            return GenerativeModelPreview(self.model_id[0])
        initialize_vertexai_params(location=self.location)
        return GenerativeModel(self.model_id[0])

    def _generate_openai(self, messages):
        """
        Generates a response using OpenAI's GPT model.
        """
        completion = self.model.chat.completions.create(
            model=self.model_id[0],
            messages=messages,
            temperature=0.6,
            top_p=0.9,
        )
        return completion.choices[0].message.content

    def _generate_vertexai(self, messages, is_grounded: Optional[bool] = False, web_search: Optional[bool] = False):
        """
        Generates a response using Claude/Gemini via VertexAI.
        """
        content = " ".join([message["content"] for message in messages])
        if "claude" in self.model_id[0]:
            response = self.model.messages.create(
                max_tokens=self.tokens,
                model=self.model_id[0],
                messages=messages,
            )
            return response.content[0].text
        elif "llama" in self.model_id[0]:
            response = self.model.chat.completions.create(
                model=self.model_id[0],
                messages=messages,
                max_tokens=self.tokens,
                temperature=self.temperature,
                # top_p=0.9,
            )
            return response.choices[0].message.content
        else:
            if is_grounded:
                if web_search:
                    tool = Tool.from_google_search_retrieval(
                        google_search_retrieval=grounding.GoogleSearchRetrieval()
                    )
                else:
                    tool = Tool.from_retrieval(
                        grounding.Retrieval(
                            grounding.VertexAISearch(
                                datastore=os.environ["VERTEXAI_DATASTORE_ID"],
                                project=VERTEXAI_PROJECT,
                                location="global",
                            )
                        )
                    )
                response = self.model.generate_content(
                    content,
                    tools=[tool],
                    generation_config=GenerationConfig(
                        temperature=1.0,
                        max_output_tokens=8192,
                        top_p=0.95,
                        # response_mime_type="application/json", #does not work
                    ),
                )
                return response
            else:
                response = self.model.generate_content(content)
                return response.text

    def _generate_huggingface(self, messages):
        """
        Generates a response using Hugging Face models.
        """
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def generate(self, messages, is_grounded: Optional[bool] = False, web_search: Optional[bool] = False):
        """
        Generates a response based on the model type.
        """
        model_type = self.model_id[0].lower()

        if "gpt-4" in model_type:
            return self._generate_openai(messages)

        elif model_type in ["claude-3-5-sonnet@20240620", "gemini-1.5-pro-002", "claude-3-5-sonnet-v2@20241022", \
                            "gemini-2.0-flash-exp", "meta/llama-3.2-90b-vision-instruct-maas"]:
            return self._generate_vertexai(messages, is_grounded, web_search)

        else:  # Hugging Face models
            return self._generate_huggingface(messages)
