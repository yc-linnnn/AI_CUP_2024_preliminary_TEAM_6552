
import transformers
import torch


original_source = './競賽訓練資料集/競賽資料集/reference/finance_pymu_cleaned'

source = './競賽訓練資料集/競賽資料集/reference/finance_pymu_cleaned_chunked_3_1000'

def is_financial_report(text):
    """
    Args:
    text (str): The text to be checked.

    Returns:
    bool: Whether the text is a financial report.

    """

    total = len(text)
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    number_count = 0
    for char in text:
        if char in numbers:
            number_count += 1

    if number_count / total > 0.3:
        return True
    else:
        return False

def title_is_in_text(title, text):
    """
    Args:
    title (str): The title to be checked.
    text (str): The text to be checked.

    Returns:
    bool: Whether the title is in the text.

    """

    if title[:3] not in text:
        return False

    m = len(title)
    n = len(text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if title[i - 1] == text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    if dp[m][n] >= 0.8 * m:
        first_match = 0
        last_match = n
        for i in range(m, 0, -1):
            if dp[i][n] <= 0.8 * m:
                last_match = i
                break
        for i in range(1, m + 1):
            if dp[i][n] >= 0.2 * m:
                first_match = i
                break
        if last_match - first_match <= m:
            return True
        return False
    else:
        return False

def llama_title_extract(text_list, chunked_text_list, ocred_stats, prompt):
    """
    Args:
    text_list (list): A list of strings, each representing a text.
    chunked_text_list (list): A list of lists of strings, each representing a chunked text.
    ocred_stats (list): A list of booleans, each representing whether the text is OCRed.
    finance_num (list): A list of strings, each representing the file number.
    prompt (str): The prompt for the Llama model.

    Returns:
    list: A list of strings, each representing a title.

    """

    model_name = 'meta-llama/Llama-3.1-8B-Instruct'

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda:0",
    )

    title_list = []

    for text, chunks, ocred in zip(text_list, chunked_text_list, ocred_stats):


        if not is_financial_report(text) or ocred:
            title_list.append('')
            continue

        title = ''
        patient = 2
        while '國巨股份' in title or '聯華電子' in title or '中國鋼鐵' in title or title == '':
            for chunk in chunks:

                instruction = prompt.replace('{text_input}', chunk)

                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": instruction},
                ]
                outputs = pipeline(
                messages,
                max_new_tokens=256,
                )
                title = outputs[0]["generated_text"][-1]['content']
                if '無法' not in title and title != '':
                    break
            patient -= 1
            if patient == 0:
                break
        
        if '無法' not in title and title != '' and title_is_in_text(title, text) and len(title) <= 70:
            title_list.append(title)
        else:
            title = ''
            title_list.append(title)

    
    return title_list
