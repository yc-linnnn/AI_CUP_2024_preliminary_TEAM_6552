
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

def get_inputs__llm_reranker(pairs, tokenizer, prompt=None, max_length=1024):

    """
    Prepare inputs for a prompt-based language model for reranking.

    Args:
    pairs (list of tuple): A list of query-passage pairs.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer of the model.
    prompt (str): The prompt for the language model.
    max_length (int): The maximum length of the input sequence.

    Returns:
    dict: A dictionary containing the input tensors.

    """

    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )



def semantic_retrieval_chunked_llm_reranker(query, source, corpus_dict, model, tokenizer, penalty_list = [], with_penalty=False):
    """
    Perform semantic retrieval using a prompt-based language model for reranking.
    
    Args:
    query (str): The query to be used for retrieval.

    source (list): A list of candidate file IDs.
    corpus_dict (dict): A dictionary containing the corpus of documents where the 
    keys are file IDs and the values are lists of chunks from corresponding file.

    model (transformers.PreTrainedModel): The language model for reranking.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer of the model.
    penalty_list (list): A list of file IDs to be penalized. They are ocred so 
    we don't have confidence in the content.
    with_penalty (bool): Whether to apply penalty to the scores.

    Returns:
    str: The file ID of the best document.
    """


    filtered_corpus = [corpus_dict[int(file)] for file in source]


    aggregated_scores = []

    file_ids = []
    pairs = []
    for i, chunks in enumerate(filtered_corpus):
        for chunk in chunks:
            pairs.append((query, chunk))
            file_ids.append(i)
        if len(chunks) == 0:
            pairs.append((query, ''))
            file_ids.append(i)
    with torch.no_grad():
        inputs = get_inputs__llm_reranker(pairs, tokenizer).to(model.device)
        outputs = model(**inputs, return_dict=True, cutoff_layers=[28])
        scores = [output[:, -1].view(-1, ).float() for output in outputs[0]]
        scores = scores[0]
        
    # aggregate by max
    for i in range(len(filtered_corpus)):
        pos = [j for j, x in enumerate(file_ids) if x == i]
        #print(f'pos: {pos}')
        scores_i = scores[pos]
        if len(scores_i) == 0:
            print(f'No scores for file {i}')
        aggregated_scores.append(scores_i.max().item())

    if with_penalty:
        for score, file in zip(aggregated_scores, source):
            if int(file) in penalty_list:
                aggregated_scores[source.index(file)] -= 1

    best_score = max(aggregated_scores)
    best_idx = aggregated_scores.index(best_score)

    # release memory
    del inputs, outputs, scores
    torch.cuda.empty_cache()

    return source[best_idx]
