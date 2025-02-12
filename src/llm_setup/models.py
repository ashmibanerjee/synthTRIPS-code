from src.llm_setup.model_init import LLMBaseClass


class Llama3(LLMBaseClass):
    """
    Initializes a Llama3 model object
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3-8B"

        super().__init__(self.model_id)


class Mistral(LLMBaseClass):
    """
    Initializes a Mistral model object
    """

    def __init__(self) -> None:
        self.model_id = "mistralai/Mistral-7B-v0.3"
        super().__init__(self.model_id)


class Gemma2(LLMBaseClass):
    """
    Initializes a Gemma2 model object
    """

    def __init__(self) -> None:
        self.model_id = "google/gemma-2-9b"
        super().__init__(self.model_id)


class Llama3Point1(LLMBaseClass):
    """
    Initializes a Llama 3.1 object
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3.1-8B"
        super().__init__(self.model_id)


class Llama3Instruct(LLMBaseClass):
    """
    Initializes a Llama 3 Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        super().__init__(self.model_id)


class MistralInstruct(LLMBaseClass):
    """
    Initializes a Mistral Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        super().__init__(self.model_id)


class Llama3Point1Instruct(LLMBaseClass):
    """
    Initializes a Llama 3.1 Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        super().__init__(self.model_id)


class Phi3SmallInstruct(LLMBaseClass):
    """
    Initializes a Phi3-Small-Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "microsoft/Phi-3-small-128k-instruct"
        super().__init__(self.model_id)


class GPT4(LLMBaseClass):
    """
    Initializes a GPT-4o-mini Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "gpt-4o-mini",
        super().__init__(self.model_id)


class GPT4o(LLMBaseClass):
    """
    Initializes a GPT-4o Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "gpt-4o",
        super().__init__(self.model_id)


class GPTo1Mini(LLMBaseClass):
    """
    Initializes a GPT-o1-mini Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "o1-mini",
        super().__init__(self.model_id)


class Claude3Point5Sonnet(LLMBaseClass):
    """
       Initializes a Claude object
       """

    def __init__(self, location=None) -> None:
        # self.model_id = "claude-3-5-sonnet@20240620",
        self.model_id = "claude-3-5-sonnet-v2@20241022",
        super().__init__(self.model_id, location="europe-west1")


class Gemini1Point5Pro(LLMBaseClass):
    """
       Initializes a Gemini Instruct object
       """

    def __init__(self, location=None) -> None:
        self.model_id = "gemini-1.5-pro-002",
        super().__init__(self.model_id, location=location)


class Gemini2Flash(LLMBaseClass):
    """
       Initializes a Gemini Flash object
       """

    def __init__(self, location=None) -> None:
        self.model_id = "gemini-2.0-flash-exp",
        super().__init__(self.model_id, location)


class Llama3Point2Vision90B(LLMBaseClass):
    """
       Initializes a Llama object
       """

    def __init__(self, location=None) -> None:
        self.model_id = "meta/llama-3.2-90b-vision-instruct-maas",
        super().__init__(self.model_id, location="us-central1")
