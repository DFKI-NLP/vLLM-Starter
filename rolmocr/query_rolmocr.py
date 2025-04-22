from openai import OpenAI
import base64
import json
import os



def encode_image(image_path):
    """
    Function which takes the path to an image as an input
    and returns the encoded picture.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ocr_page_with_rolm(img_base64, client, model):
    """
    Function which takes an encoded picture as input as well
    as the pre-defined client and model-name.

    Prompts to the model and returns the answer.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Return the plain text representation of this document as if you were reading it naturally.",
                    },
                ],
            }
        ],
        temperature=0.2,
        max_tokens=4096
    )
    return response.choices[0].message.content

# Example usage          

def run():

    # Connect to your own vLLM server
    client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")

    model = "reducto/RolmOCR"

    with open('./testdaten/data.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)

    for document in data:
        ocr_results = []
        image_filenames = document['pages']

        for image in image_filenames:
            for file in os.listdir('./testdaten'):
                if file == image:
                    img_b64 = encode_image(file)
                    text = ocr_page_with_rolm(img_b64, client, model)
                    ocr_results.append(text)
        document["rolm_ocr"] = ocr_results

    with open("./testdaten/data_updated_rolmOCR.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

run()
