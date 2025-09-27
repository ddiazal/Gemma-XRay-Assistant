from torch import Tensor
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from typing import Any, Dict, List

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="mpc",
)

class Assistant:
    def __init__(self) -> None:
        self.model = model
        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(self, image) -> str:
        im_inputs: Tensor = self.get_inputs(image)

        response_tensor: Tensor = self.get_inference(inputs=im_inputs, max_tokens=200)

        generated_report: str = self.processor.decode(response_tensor, skip_special_tokens=True)

        return generated_report


    @staticmethod
    def get_message(image) -> List[Dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "# Identity\n"
                            "You are an expert radiologist assistant that excels at succinctly describing "
                            "the findings for a X-ray image.\n\n"
                            "# Task\n"
                            "Your task is to accurately describe the findings for a X-ray image.\n\n"
                            "# Instructions\n"
                            "Go step by step throughout the following instructions:\n"
                            "1- Consider all details in the image.\n"
                            "2- Use a friendly but sincere tone.\n"
                            "3- Use an adequate technical level for average people.\n"
                            "4- Provide a concise description of the findings and do not say anything else.\n\n"
                            "# Output Format\n"
                            "Provide your description inside <report></report> tags.\n\n"
                            "# Output\n"
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this X-ray"},
                    {"type": "image", "image": image}
                ]
            }
        ]

        return messages

    def get_inference(self, inputs: Any, max_tokens: int = 200) -> Tensor:
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            generation = generation[0][input_len:]

        return generation

    def get_inputs(self, image) -> Tensor:
        messages = self.get_message(image)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        return inputs

    def generation_accuracy(self, image) -> float:
        generation = self.get_inference(image)