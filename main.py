
import os
import json
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from Preprocess.data_preprocess import clean_insurance_text, clean_finance_text, chunking

from Preprocess.pdf_preprocess import read_pdf_pdfplumber_with_ocr, read_pdf_pdfplumber

from Preprocess.llama_title_extraction import llama_title_extract, title_is_in_text

from Model.retrieve import semantic_retrieval_chunked_llm_reranker



import torch



if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--use_pre_built_dataset', action='store_true', help='是否使用預建立的資料集, set to True to save time. If not set, it will go through the whole process of pdf reading, data cleaning, ocr...')  # 是否使用預建立的資料集
    parser.add_argument('--use_pre_built_document_titles', action='store_true', help='是否使用預建立的文件標題, only used when use_pre_built_dataset is set to False. If set, a llama model will be used for extracting titles of document. Recommended to omit for saving time.')  # 是否使用預建立的文件標題  

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    if not args.use_pre_built_dataset:

        source_path_insurance = os.path.join(args.source_path, 'insurance')

        insurance_files = os.listdir(source_path_insurance)

        insurance_file_num = [int(file.split('.')[0]) for file in insurance_files]

        insurance_text = [read_pdf_pdfplumber(os.path.join(source_path_insurance, file)) for file in tqdm(insurance_files, desc='Reading insurance pdfs')]

        insurance_text = [clean_insurance_text(text) for text in tqdm(insurance_text, desc='Cleaning insurance text')]

        insurance_chunks = [chunking(text, 1000) for text in tqdm(insurance_text, desc='Chunking insurance text')]

        corpus_dict_insurance = {i: chunks for i, chunks in zip(insurance_file_num,insurance_chunks)}


        source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑

        finance_files = os.listdir(source_path_finance)

        finance_file_num = [int(file.split('.')[0]) for file in finance_files]

        finance_text = [read_pdf_pdfplumber_with_ocr(os.path.join(source_path_finance, file)) for file in tqdm(finance_files, desc='Reading finance pdfs')]

        finance_text, ocred_stats = map(list, zip(*finance_text))


        penalty_list = [int(file) for file, ocred in zip(finance_file_num, ocred_stats) if ocred]

        finance_text = [clean_finance_text(text) for text in tqdm(finance_text, desc='Cleaning finance text')]

        finance_chunks = [chunking(text, 500) for text in tqdm(finance_text, desc='Chunking finance text')]



        if args.use_pre_built_document_titles:
            print('Using pre-built document titles')

            titles = []

            for num in tqdm(finance_file_num, desc='Extracting finance titles'):
                #print(f'Processing file {num}...')
                dir = './datasets/finance_llama_titles'
                title_path = os.path.join(dir, f'{num}.txt')
                try:
                    with open(title_path, 'r') as f:
                        title = f.read()
                except:
                    title = ''
                titles.append(title)

        else:
            print('Building document titles')
            finance_chunks_1000 = [chunking(text, 1000) for text in tqdm(finance_text, desc='Chunking finance text for llama title extraction')]

            prompt_path = './datasets/extract_title_prompt.txt'
            with open(prompt_path, 'r') as f:
                prompt = f.read()
            titles = llama_title_extract(finance_text, finance_chunks_1000, ocred_stats, prompt)
            
        finance_chunks_with_titles = [
            [
                title + '\n' + chunk if title != '' and len(title) <= 70 and not title_is_in_text(title, chunk) 
                else chunk 
                for chunk in chunk_list
            ]
        for title, chunk_list in tqdm(zip(titles, finance_chunks), desc='Adding titles to finance chunks')
        ]

        corpus_dict_finance = {i: chunks for i, chunks in zip(finance_file_num, finance_chunks_with_titles)} 

    else:
        insurance_chunks_path = "./datasets/insurance_chunks.json"
        finance_chunks_path = "./datasets/finance_chunks.json"

        with open(insurance_chunks_path, 'r', encoding='utf8') as f:
            corpus_dict_insurance = json.load(f)
        
        with open(finance_chunks_path, 'r', encoding='utf8') as f:
            corpus_dict_finance = json.load(f)

        # cast the key to int
        corpus_dict_insurance = {int(key): value for key, value in corpus_dict_insurance.items()}
        corpus_dict_finance = {int(key): value for key, value in corpus_dict_finance.items()}

        penalty_list_path = "./datasets/penalty_list.json"
        with open(penalty_list_path, 'r', encoding='utf8') as f:
            penalty_list = json.load(f)

    model_name = 'BAAI/bge-reranker-v2-minicpm-layerwise'

    if model_name == 'BAAI/bge-reranker-large':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-minicpm-layerwise', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-minicpm-layerwise', trust_remote_code=True, torch_dtype=torch.bfloat16)

    model = model.to('cuda')
    model.eval()

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for i, q_dict in enumerate(qs_ref['questions']):
        print(f'Processing question {i+1}/{len(qs_ref["questions"])}')
        if q_dict['category'] == 'finance':
            retrieved = semantic_retrieval_chunked_llm_reranker(q_dict['query'], q_dict['source'], corpus_dict_finance, model, tokenizer,  penalty_list, with_penalty=True)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        elif q_dict['category'] == 'insurance':
            retrieved = semantic_retrieval_chunked_llm_reranker(q_dict['query'], q_dict['source'], corpus_dict_insurance,  model, tokenizer)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: [str(value)] for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = semantic_retrieval_chunked_llm_reranker(q_dict['query'], q_dict['source'], corpus_dict_faq,  model, tokenizer)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
