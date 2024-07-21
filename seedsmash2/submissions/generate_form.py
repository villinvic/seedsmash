import inspect
from dataclasses import fields
from typing import Type

from bot_config import BotConfig

def generate_file_template(cls):
    """
    Generate a user-friendly file template from the dataclass.
    """
    template_lines = []
    x = "  | "
    xs = "  * "


    for docline in cls.__doc__.split("\n"):
        template_lines.append(docline)

    template_lines.append("="*90+"\n")

    for name, doc in cls.__field_docs__.items():
        doc = doc.rstrip("\n ").replace('\n', '\n' +x)

        template_lines.append(f"[INPUT] {name} = {getattr(cls, name)}")
        template_lines.append(xs + f"{doc}\n"+xs )

    return "\n".join(template_lines)


def write_template_to_file(filename, content):
    """
    Write the generated template content to a file.
    """
    with open(filename, 'w') as f:
        f.write(content)

def load_filled_form(filename):
    """
    Load the filled form from a file and create a namedtuple object.
    Long term, we should use an actual form
    """
    obj = BotConfig()
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("[INPUT] "):
            val = line.lstrip("[INPUT] ")
            attribute, value = val.split("=")
            attribute = attribute.strip(" ")
            value = value.strip(" ")

            if value == "":
                continue

            if hasattr(obj, attribute):
                # TODO: take care of types
                setattr(obj, attribute, value)
    return obj

if __name__ == "__main__":
    template_content = generate_file_template(BotConfig)
    write_template_to_file("your_bot_name.txt", template_content)

    #print(load_filled_form("your_bot_name.txt"))