import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


class Qwen3_VL:
    def __init__(self, model_path, args):
        super().__init__()
        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self, messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role": "system", "content": messages["system"]})
        if "image" in messages:
            new_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": messages["image"]},
                        {"type": "text", "text": messages["prompt"]},
                    ],
                }
            )
        elif "images" in messages:
            content = []
            for i, image in enumerate(messages["images"]):
                content.append({"type": "text", "text": f"<image_{i + 1}>: "})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": messages["prompt"]})
            new_messages.append({"role": "user", "content": content})
        else:
            new_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": messages["prompt"]}],
                }
            )
        messages = new_messages
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.llm.device)

        return inputs

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            generation = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
            )
            generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def generate_outputs(self, messages_list):
        res = []
        for messages in tqdm(messages_list):
            result = self.generate_output(messages)
            res.append(result)
        return res
