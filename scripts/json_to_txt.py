import json

def convert_txt_to_json(txt_file, json_file):
    """
    Validate and convert a JSON-like .txt file into a proper .json file.

    Args:
        txt_file (str): Path to the .txt file.
        json_file (str): Path to save the .json file.
    """
    try:
        # Read the .txt file
        with open(txt_file, "r", encoding="utf-8") as file:
            txt_content = file.read()

        # Replace single trailing commas before closing braces/brackets
        txt_content = txt_content.replace(",\n}", "\n}")
        txt_content = txt_content.replace(",\n]", "\n]")

        # Try to load it as JSON
        expanded_terms = json.loads(txt_content)

        # Save as proper JSON
        with open(json_file, "w", encoding="utf-8") as json_out:
            json.dump(expanded_terms, json_out, indent=4, ensure_ascii=False)
        
        print(f"Successfully converted to {json_file}")

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print("Ensure the .txt file uses valid JSON-like syntax.")
    except Exception as e:
        print(f"Unexpected Error: {e}")

# File paths
txt_file = "expanded_terms.txt"  # Replace with your .txt file path
json_file = "expanded_terms.json"  # Path to save the converted JSON

# Convert the file
convert_txt_to_json(txt_file, json_file)
