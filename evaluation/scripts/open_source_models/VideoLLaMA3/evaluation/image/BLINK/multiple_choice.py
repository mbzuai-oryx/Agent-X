from openai import OpenAI
import os
import time
import openai
from openai import AzureOpenAI

client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
    )

def get_chat_response(promot, n=1, patience=10000000,
 sleep_time=0):
    messages = [
        {"role": "user", "content": promot},
    ]
    # print("I am here")
    while patience > 0:
        patience -= 1
        try:
            response = interaction(client, messages)
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                # reduce input prompt and keep the tail
                new_size = int(len(promot) * 0.9)
                new_start = len(promot) - new_size
                promot = promot[new_start:]
                messages = [
                    {"role": "user", "content": promot},
                ]
                
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def init():
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
    )

    return client


def interaction(client, message_text):
    completion = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYNAME"),
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion

def build_prompt(question, options, prediction):
    """
    Builds the prompt for the GPT-3.5 turbo model to match an answer with several options of a single-choice question.

    If the GPT-3.5 model is unable to find a match, it will output (Z).
    Also, if the original prediction does not clearly lean towards any of the options, it will output (Z).

    Parameters:
    - question: String, the question.
    - options: String, the options. E.g. ['(A)', '(B)']
    - prediction: String, the answer. E.g. '(B)'
    """
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (Z)"
        "If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (Z)"\
        "Your should output one of the choices, (A),(B),(C),(D),(E) (if they are valid options), or (Z)\n"
        "Example 1: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n"
        "Example 2: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        "Example 3: \n"
        "Question: Which point is corresponding to the reference point?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer:The reference point (REF) on the first image is at the tip of the pot, which is the part used to Poke if the pots were used for that action. Looking at the second image, we need to find the part of the object that would correspond to poking.\n(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n(B) Point B is at the bottom of the spoon, which is not used for poking.\n(C) Point C is on the side of the pspoonot, which is not used for poking.\n(D) Point D is at the tip of the spoon, which is not used for poking.\n\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n"
        "Example 4: \n"
        "Question: {}?\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)


def match_multiple_choice(question, options, prediction):
    prompt = build_prompt(question, options, prediction)
    retry_limit = 10
    
    for retry in range(retry_limit):
        try:
            extraction = get_chat_response(prompt, patience=10)
            return extraction
        except Exception as e:
            time.sleep(1)
    return '(Z) Failed to get multiple choice'

if __name__ == "__main__":
    client = init()
    print(match_multiple_choice("Which point is corresponding to the reference point?\nSelect from the following choices.", "(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D", "The reference point (REF) on the first image is located at the tip of the spatula, which is the part of the tool typically used to scrape surfaces. To find the corresponding point for the action \"Scrape\" on the second image, we need to identify the part of the tool that would be used in a similar manner.\n\nLooking at the second image:\n\n(A) Point A is on the side edge of the blade, which is not typically used for scraping.\n(B) Point B is on the top edge of the blade, which is also not used for scraping.\n(C) Point C is on the handle, which is not the scraping part but rather the part you hold.\n(D) Point D is on the label near the handle, which is also not relevant to the scraping action.\n\nNone of the labeled points correspond to the scraping edge of the tool in the second image. However, the closest equivalent part for scraping would be the unmarked edge opposite to Point A, which is the flat, sharp edge of the blade used for scraping. Since none of the provided choices accurately represent the scraping edge, none of the labeled points (A, B, C, D) are correct. The correct corresponding point for scraping is not marked on the second image."))
