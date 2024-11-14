
import re


def clean_finance_text(finance_text: str):
    """
    Cleans and processes finance-related text by removing unnecessary whitespace, 
    redundant characters, and unwanted lines.

    Args:
    finance_text (str): The text to be cleaned.

    Returns:
    str: The cleaned text, with repeated spaces reduced, and certain lines removed.
    """
    lines = finance_text.splitlines(keepends=True)

    lines = [line.strip() for line in lines]

    # delete repeated spaces using regex
    lines = [re.sub('\s+', ' ', line) for line in lines]

    #delete lines that start with '~' and end with '~' or start with '-' and end with '-'
    lines = [line for line in lines if not (line.startswith('~') and line.endswith('~')) and not (line.startswith('-') and line.endswith('-')) or len(line) > 10]
        
    lines = [line for line in lines if len(line) > 1]

    content = '\n'.join(lines)


    return content


def clean_insurance_text(insurance_text: str):

    """
    Cleans and processes insurance-related text by removing unnecessary whitespace,
    redundant characters, and unwanted lines. Explicitly crafted rules are used for
    adjusting newlines characters.

    Args:
    insurance_text (str): The text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    
    # Define starters and enders for line filtering
    starter = ['第', '【']

    end = ['。', '？', '！', '；', '：', '」', '』', '”', '』', '）', ')', '】', ']']
    # delete the newlines if this line didn't start with the starter and end with the end

    #lines = insurance_text.split('\n')

    lines = insurance_text.splitlines(keepends=True)

    new_lines = []
    for i, line in enumerate(lines):
        line = line[:-1]
        line = line.strip()
        if line.startswith('第') and line.endswith('頁') and len(line) < 20:
            continue
        if ' / ' in line and len(line) < 10:
            continue
        if '銷售日期' in line and '年' in line and '月' in line and '日' in line and len(line) < 30:
            continue
        if len(line) ==0:
            continue
        if len(new_lines) == 0:
            new_lines.append(line+'\n')
            continue
        if '_' in line or ('月' in line and '日' in line and len(line) < 20):
            new_lines.append(line+'\n')
            continue
        if line[0] not in starter and line[-1] not in end:
            new_lines.append(line)
        elif line[0] in starter and len(line) < 2:
            new_lines.append(line)
        else:
            new_lines.append(line+'\n')

    new_text = ''.join(new_lines)


    return new_text

def chunking(text: str, chunk_size: int):

    """
    Splits a text into chunks of a specified size, with overlap between chunks.

    Args:
    text (str): The text to be split into chunks.
    chunk_size (int): The upper size limit of each chunk.

    Returns:
    list: A list of strings, each representing a chunk of the original text.
    """

    lines = text.splitlines(keepends=True)
    chunks = []
    
    chunk = ''
    

    chunks = []
    
    for i, line in enumerate(lines):
        while len(line)  > chunk_size:
            chunks.append(line[:chunk_size])
            line = line[chunk_size:]
        if len(chunk) + len(line) > chunk_size:
            chunks.append(chunk)
            chunk = ''
        chunk += line
    if len(chunk) > 0 and chunk != '':
        chunks.append(chunk)
    overlap_chunks = []
    for i,chunk in enumerate(chunks):
        if i == 0:
            continue
        new_chunk =''
        prev_lines = chunks[i-1].split('\n')
        prev_lines = [line +'\n' for line in prev_lines if line != '']
        while len(new_chunk)< (chunk_size)//2 and len(prev_lines) > 0:
            new_chunk = prev_lines.pop(-1) + new_chunk
        next_lines = [line + '\n' for line in chunks[i].split('\n') if line != '']
        k = 0
        flag = False
        while len(new_chunk) + len(next_lines[k]) < chunk_size:
            flag = True
            new_chunk += next_lines[k]
            k+=1
            if k == len(next_lines):
                break
        if flag:
            overlap_chunks.append(new_chunk)
        
        
    # add overlap chunks to chunks
    chunks = chunks + overlap_chunks
    # delete newlines and spaces in each chunk
    chunks = [newline for newline in chunks if len(newline)>chunk_size//3]
    chunks = [new_line.replace(' ', '').replace('\n', ' ') for new_line in chunks]

    return chunks