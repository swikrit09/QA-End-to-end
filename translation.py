from languages import foreign_languages, indian_languages
import os
import requests
from dotenv import load_dotenv

load_dotenv()
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
RAPID_API_HOST = os.getenv("RAPID_API_HOST")
BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID")
BHASHINI_ULCA = os.getenv("BHASHINI_ULCA")
PIPELINE_ID = os.getenv("PIPELINE_ID")

print(BHASHINI_ULCA)

bhashini_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
google_translate_url = "https://google-translator9.p.rapidapi.com/v2"


def get_languages():
    return {"indian_languages":indian_languages, "foreign_languages": foreign_languages}

def get_language_names():
    return foreign_languages.keys()

def bhashini_translation(source_language,target_language,content):
    payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                            "targetLanguage": target_language
                        }
                    }
                }
            ],
            "pipelineRequestConfig": {
                "pipelineId": PIPELINE_ID
            }
        }

    headers = {
        "Content-Type": "application/json",
        "userID": BHASHINI_USER_ID,
        "ulcaApiKey": BHASHINI_ULCA
    }

    response = requests.post(bhashini_url, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        service_id = response_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        compute_payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                            "targetLanguage": target_language
                        },
                        "serviceId": service_id
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": content
                    }
                ],
                "audio": [
                    {
                        "audioContent": None
                    }
                ]
            }
        }

        callback_url = response_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]

        headers2 = {
            "Content-Type": "application/json",
            response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]:
                response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        }

        compute_response = requests.post(callback_url, json=compute_payload, headers=headers2)
        print(compute_response)
        if compute_response.status_code == 200:
            compute_response_data = compute_response.json()
            translated_content = compute_response_data["pipelineResponse"][0]["output"][0]["target"]
            return {
                "status_code": 200,
                "message": "Translation successful",
                "translated_content": translated_content
            }
        else:
            return {
                "status_code": compute_response.status_code,
                "message": "Error in translation",
                "translated_content": None
            }
    else:
        return {
            "status_code": response.status_code,
            "message": "Error in translation request",
            "translated_content": None
        }
        
            
def google_translate_translation(source_language,target_language,content):
    
    payload = {
        "q": content,
        "source": source_language,
        "target": target_language,
        "format": "text"
    }
    headers = {
        "x-rapidapi-key": RAPID_API_KEY,
        "x-rapidapi-host": RAPID_API_HOST,
        "Content-Type": "application/json"
    }

    response = requests.post(google_translate_url, json=payload, headers=headers)
    data = response.json()
    if response.status_code == 200 and "data" in data and "translations" in data["data"]:
        translated_content = data["data"]["translations"][0]["translatedText"]
        return {
            "status_code": 200,
            "message": "Translation successful",
            "translated_content": translated_content
        }
    else:
        return {
            "status_code": response.status_code,
            "message": "Error in translation request",
            "translated_content": None
        } 


def translate(source_language,target_language,content):
    if source_language is None or target_language is None:
        return {"status_code": 400, "message": "Invalid source or target language", "translated_content": None}
    
    content = content
    source_language = (
        indian_languages.get(source_language)
        if source_language in indian_languages
        else foreign_languages.get(source_language)
    )
    target_language = (
        indian_languages.get(target_language)
        if target_language in indian_languages
        else foreign_languages.get(target_language)
    )

    # Condition for Google API

    return google_translate_translation(source_language,target_language,content)

# Example usage
# print(translate("English", "Spanish", "Hello"))

    


