import json
import logging

import backoff
import dspy
import requests
import threading
from typing import Optional, Literal, Any, List

from dsp import AzureOpenAI

LM_LRU_CACHE_MAX_SIZE = 3000

def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end

def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end

def _inspect_history(lm, n: int = 1):
    """Prints the last n prompts and their completions."""
    for item in lm.history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        print("\n\n\n")
        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            print(msg["content"].strip())
            print("\n")
        print(_red("Response:"))
        print(_green(outputs[0].strip()))
        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end=""))
    print("\n\n\n")



class OpenAIModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        # Log the token usage from the OpenAI API response.
        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions

ERRORS = (
    requests.exceptions.RequestException,
    Exception,  # Fallback
)

def backoff_hdlr(details):
    print(f"Backing off {details['wait']} seconds after {details['tries']} tries.")

def giveup_hdlr(e):
    return False

# api_base: str = "https://api.groq.com/openai/v1",
class GroqModel(dspy.OpenAI):
    def __init__(
            self,
            model: str = "llama3-70b-8192",
            api_key: Optional[str] = None,
            api_base: str = None,
            **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        if not self.api_key:
            raise ValueError(
                "Groq API key must be provided either as an argument or as an environment variable GROQ_API_KEY"
            )

    def log_usage(self, response):
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        usage = {
            self.model: {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def _create_completion(self, prompt: str, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        kwargs.pop("logprobs", None)
        kwargs.pop("logit_bias", None)
        kwargs.pop("top_logprobs", None)
        if "n" in kwargs and kwargs["n"] != 1:
            raise ValueError("Groq API only supports N=1")
        if kwargs.get("temperature", 1) == 0:
            kwargs["temperature"] = 1e-8
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        for message in data["messages"]:
            message.pop("name", None)
        response = requests.post(
            f"{self.api_base}/chat/completions", headers=headers, json=data
        )
        response.raise_for_status()
        return response.json()

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        response = self._create_completion(prompt, **kwargs)
        self.log_usage(response)
        choices = response["choices"]
        completions = [choice["message"]["content"] for choice in choices]
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        }
        self.history.append(history)
        return completions

class OllamaClient(dspy.OllamaLocal):
    """A wrapper class for dspy.OllamaClient."""

    def __init__(self, model, port, url="http://localhost", **kwargs):
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        super().__init__(model=model, base_url=f"{url}:{port}", **kwargs)
        self.kwargs = {**self.kwargs, **kwargs}


class AzureOpenAIModel(dspy.AzureOpenAI):
    """A wrapper class of Azure OpenAI endpoint.

    Note: param::model should match the deployment_id on your Azure platform.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_version: str,
        model: str,
        api_key: str,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        super().__init__(model=model)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = model
        self.provider = "azure"
        self.model_type = model_type

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def basic_request(self, prompt: str, **kwargs) -> Any:
        kwargs = {**self.kwargs, **kwargs}

        try:
            if self.model_type == "chat":
                messages = [{"role": "user", "content": prompt}]

                response = self.client.chat.completions.create(
                    messages=messages, **kwargs
                )
            else:
                response = self.client.completions.create(prompt=prompt, **kwargs)

            self.log_usage(response)

            history_entry = {
                "prompt": prompt,
                "response": dict(response),
                "kwargs": kwargs,
            }
            self.history.append(history_entry)

            return response

        except Exception as e:
            logging.error(f"Error making request to Azure OpenAI: {str(e)}")
            raise

    def _get_choice_text(self, choice: Any) -> str:
        """Extract text from a choice object based on model type."""
        if self.model_type == "chat":
            return choice.message.content
        return choice.text

    def log_usage(self, response):
        """Log the total tokens from response."""
        usage_data = response.usage
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.prompt_tokens
                self.completion_tokens += usage_data.completion_tokens

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.model: {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[str]:
        """Get completions from Azure OpenAI.

        Args:
            prompt: The prompt to send to the model
            only_completed: Only return completed responses
            return_sorted: Sort completions by probability (not implemented)
            **kwargs: Additional arguments to pass to the API

        Returns:
            List of completion strings
        """
        response = self.basic_request(prompt, **kwargs)

        choices = response.choices
        completed_choices = [c for c in choices if c.finish_reason != "length"]

        if only_completed and completed_choices:
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]

        return completions