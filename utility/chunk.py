from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from utility.embedding import get_embedding, lora_fine_tune
from utility.pg import insert_chunks, get_training_samples

CUR_PATH = Path(__file__).resolve().parent

def test():
    print(f"chunk.py模块已经启动")

def split_text_into_chunks(long_text: str):
    '''将长文本拆分为多个块，每个块的长度不超过500个字符，块之间有80个字符的重叠。'''
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "。", "！", "？",  
            "；", "，",     
            " ", "" 
        ],
        chunk_size=500,      
        chunk_overlap=80,   
        length_function=len, 
        keep_separator='end',
    )
    chunks = text_splitter.split_text(long_text)
    return chunks

def foreach_chunk_source():
    '''从所在目录的父目录的chunksource文件夹中 chunk所有txt文件的内容
    并获取对应的embedding 以及插入到数据库中'''

    chunk_file_path = os.path.join(CUR_PATH.parent, "chunksource")
    records = []
    for filename in os.listdir(chunk_file_path):
        if ".txt" in filename:
            title, _ = filename.rsplit('.', 1) # 把filename按从右往左的第一个.字符进行分割
            file_path = os.path.join(chunk_file_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = split_text_into_chunks(content)
                embeds = get_embedding(chunks)
                for i, chunk in enumerate(chunks):
                    record = (title, chunk, embeds[i])
                    print(f"title = {title}, chunk = {len(chunk)}, " + 
                          f"embedding = {type(embeds[i])}")
                    records.append(record)
    insert_chunks(records)

def fine_tune_model():
    '''对embedding模型进行lora微调'''
    lora_fine_tune(get_training_samples())



if __name__ == "__main__":
    foreach_chunk_source()
