import csv
from itertools import islice

import torch
import fastchat.model
from transformers import AutoTokenizer
from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

def load_conversation_template(template_name: str):

    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'llama-2':
        conv_template.system_message = "You are a helpful assistant."
    if conv_template.name == 'mistral':
        MISTRAL_SYSPROMPT = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
        conv_template.system_message = MISTRAL_SYSPROMPT
    return conv_template


class PromptManager:
    def __init__(self, *, tokenizer, conv_template, instruction: str, target: str, adv_string: str):
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

        self._user_role_slice = None
        self._goal_slice = None
        self._control_slice = None
        self._assistant_role_slice = None
        self._target_slice = None
        self._loss_slice = None

    @property
    def _is_llama_2_format(self):
        if self.conv_template.name == 'llama-2':
            return True
        if self.conv_template.name == 'mistral':
            return True
        return False

    # TODO：add more model prompt templates.
    # TODO: rewrite numbers as constants.
    def get_prompt(self, adv_string: str = None) -> str:
        final_prompt_str = None

        if adv_string is not None:
            self.adv_string = adv_string

        if self._is_llama_2_format:

            self.conv_template.messages = []
            # The addition and subtraction operations are due to fschat library's implementation.
            # See: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
            test = "a"
            self.conv_template.append_message(self.conv_template.roles[0], test)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks) - 2)

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, len(toks) - 1)

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._adv_slice = slice(self._goal_slice.stop, len(toks) - 1)

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._adv_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            final_prompt_str = self.conv_template.get_prompt()
            toks = self.tokenizer(final_prompt_str).input_ids

            # skip </s><s>
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        elif self.conv_template.name == "gemma":
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # skip `self.sep` (<end_of_turn>\n)
            self._goal_slice = slice(self._user_role_slice.stop, len(toks) - 2)

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._adv_slice = slice(self._goal_slice.stop, len(toks) - 2)

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._adv_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            final_prompt_str = self.conv_template.get_prompt()
            toks = self.tokenizer(final_prompt_str).input_ids

            # skip `self.sep` (<end_of_turn>\n)
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        elif self.conv_template.name == "qwen-7b-chat":

            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # skip `self.sep + "\n"` (｜im_end｜)
            self._goal_slice = slice(self._user_role_slice.stop, len(toks) - 2)

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._adv_slice = slice(self._goal_slice.stop, len(toks) - 2)

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._adv_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            final_prompt_str = self.conv_template.get_prompt()
            toks = self.tokenizer(final_prompt_str).input_ids

            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        elif self.conv_template.name == "Phi-3-mini-128k-instruct":
            raise NotImplementedError("Error, fastchat does not implement Phi-3-mini-128k-instruct")

        if final_prompt_str is None:
            # 简单格式的 fallback
            separator = ' ' if self.instruction else ''
            parts = [p for p in [self.instruction, self.adv_string, self.target] if p]
            final_prompt_str = separator.join(parts)

            # 设置基本切片
            toks = self.tokenizer(final_prompt_str).input_ids
            self._user_role_slice = slice(0, 0)
            self._goal_slice = slice(0, len(self.tokenizer(self.instruction).input_ids) if self.instruction else 0)
            self._adv_slice = slice(self._goal_slice.stop, self._goal_slice.stop + len(
                self.tokenizer(self.adv_string).input_ids) if self.adv_string else self._goal_slice.stop)
            self._assistant_role_slice = slice(self._adv_slice.stop, self._adv_slice.stop)
            self._target_slice = slice(self._adv_slice.stop, len(toks))
            self._loss_slice = slice(self._adv_slice.stop, len(toks))

            print(f"WARNING: Using fallback simple prompt format for {self.conv_template.name}")

        self.conv_template.messages = []
        return final_prompt_str

    def update_affirmation(self, new_target):
        self.target = new_target

    def get_input_ids(self, adv_string: str = None) -> torch.Tensor:
        prompt_str = self.get_prompt(adv_string=adv_string)
        #print(prompt_str + "\n")
        toks = self.tokenizer(prompt_str).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])
        return input_ids

    def get_goal_slice(self):
        return self._goal_slice

    def get_loss_slice(self):
        return self._loss_slice

    def get_adv_slice(self):
        return self._adv_slice

    def get_target_slice(self):
        return self._target_slice

    def get_assistant_role_slice(self):
        return self._assistant_role_slice


def main():
    def decode_slice(slice_obj):
        return tokenizer.decode(input_ids[slice_obj], skip_special_tokens=False)

    model_path = "./models/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    conv_template = load_conversation_template("mistral")

    dataset_path = "./data/harmful_behaviors.csv"
    test = 1
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        target_row = next(islice(reader, test, test+1), None)

    manager = PromptManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=target_row[0],
        target=target_row[1],
        adv_string= "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    )

    input_ids = manager.get_input_ids()
    print(f"\ninput_ids Shape:{input_ids.shape}")
    print(f"input_ids content:{input_ids.tolist()}")

    print(f"\nslices::")
    print(f"User Role Slice:{manager._user_role_slice}")
    print(f"Goal Slice:{manager._goal_slice}")
    print(f"Control Slice:{manager._adv_slice}")
    print(f"Assistant Role Slice:{manager._assistant_role_slice}")
    print(f"Target Slice:{manager._target_slice}")
    print(f"Loss Slice:{manager._loss_slice}\n")

    print(f"all inputs:{tokenizer.decode(input_ids, skip_special_tokens=False)}")
    print(f"User Role Slice content:{decode_slice(manager._user_role_slice)}")
    print(f"Goal Slice content:{decode_slice(manager._goal_slice)}")
    print(f"Control Slice content:{decode_slice(manager._adv_slice)}")
    print(f"Assistant Role Slice content:{decode_slice(manager._assistant_role_slice)}")
    print(f"Target Slice content:{decode_slice(manager._target_slice)}")
    print(f"Loss Slice content:{decode_slice(manager._loss_slice)}")

if __name__ == "__main__":
    main()