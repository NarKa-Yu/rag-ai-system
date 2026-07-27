from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
import os
import uvicorn
from utility.embedding import get_embedding, lora_fine_tune
from utility.pg import (get_similar_chunks, make_samples, clear_samples, get_training_samples, 
                        reset_embedding)
from pathlib import Path

app = FastAPI(title="LLM API")

# 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化OpenAI异步客户端
client = AsyncOpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('BASE_URL'), 
)
MODEL_NAME = os.getenv('MODEL_NAME')


@app.get("/api/check_chunks")
async def check_chunks(request: Request):
    data = dict(request.query_params) # 读取GET请求PATH内的键值对
    top_k = data.get("top_k", 5)
    query = data.get("query", "").strip()
    embed = get_embedding([query])
    chunks = get_similar_chunks(embed[0], top_k)
    result = []
    for id, title, chunk, similar in chunks:
        result.append({'id': id, 'title': title, 'chunk': chunk, 'similar': similar})
    return result

@app.get("/api/check_dataset")
async def check_dataset():
    '''查看当前训练集'''
    data = get_training_samples()
    return data


@app.get("/api/start_lora_train")
async def start_lora_train():
    '''开始lora微调训练'''
    try:
        lora_fine_tune(get_training_samples())
        return {'state': 1, 'description': 'train complete.'}
    except Exception as e:
        return {'state': 0, 'description': f'error: {e}.'}

@app.get("/api/reset_embedding")
async def reset_embed():
    '''把documents数据库里的chunk根据当前embedding模型重置其embed字段'''
    def get_single_emb(chunk: str) -> list[float]:
        return get_embedding([chunk])[0]
    reset_embedding(embed_func=get_single_emb)
    


@app.get("/api/reset_sample")
async def reset_sample():
    '''清空当前samples表'''
    clear_samples()

@app.post("/api/chat")
async def chat_stream(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        return {"error": "query 不能为空"}

    async def generate():
        try:
            yield 'RAG检索中...\n\n'
            embed = get_embedding([query])
            chunks = get_similar_chunks(embed[0], top_k=4)
            rag_prompt = ''
            for id, title, chunk, similar in chunks:
                if similar > 0.5:
                    if len(chunk) > 50:
                        cur_info = f'标题为:{title} \n\n' + f'相关内容为:{chunk[:50]}...\n\n'
                    else:
                        cur_info = f'标题为:{title} \n\n' + f'相关内容为:{chunk}...\n\n'
                    cur_info += '----------------------------------------------\n\n'
                    rag_prompt += f'标题为:{title} \n' + f'相关内容为:{chunk}'
                    yield cur_info

            if len(rag_prompt) == 0:
                rag_prompt = '关于该提问，没有任何相关的资料可供参考，诚实告诉user无法准确回答'

            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个有帮助的助手。" + 
                     f"根据以下可以依据的材料回答用户，可进行一定程度的扩展补充: {rag_prompt}"},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=0.6,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content

        except Exception as e:
            yield f"\n[错误] {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/api/make_sample")
async def make_sample(request: Request):
    '''构造样本集'''
    data = await request.json()
    make_samples(data)



# react构建项目的静态文件资源请求
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    static_dir = Path("static")
    if full_path and not full_path.startswith("static"):
        file_path = static_dir / "index.html"
    else:
        file_path = static_dir / full_path if full_path else static_dir / "index.html"
    
    if file_path.exists():
        return FileResponse(file_path)
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)