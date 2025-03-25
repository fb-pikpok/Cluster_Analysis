import helper.utils as utils
from helper.utils import api_settings, logger
from langchain.prompts import PromptTemplate

prompt_template_testcases = PromptTemplate.from_template(
'''
Please return Hello World
'''
)

def test_configure_api():
    utils.configure_api('gpt-4o-mini')
    try:
        prompt_sentiment = prompt_template_testcases.format()
        response = api_settings["client"].chat.completions.create(
            model=api_settings["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_sentiment},
            ],
            max_tokens=25,
        )
        utils.track_tokens(response)
        message = response.choices[0].message.content.strip()
        logger.info(f'Sucessfully connected to OpenAI API')
        logger.info(f'prompt_tokens: {utils.prompt_tokens}, completion_tokens: {utils.completion_tokens}... The tokens may NOT be 0 !')
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {e}")
        raise



if __name__ == '__main__':
    test_configure_api()
