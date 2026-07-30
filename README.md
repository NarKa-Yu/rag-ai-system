# RAG AI System

基于RAG系统的AI问答以及Embeeding模型Lora微调、数据库构建全栈项目：

本Demo采用中国国务院政策文件库的"人工智能"检索词相关的政策文件作为构建RAG知识库来源(https://sousuo.www.gov.cn/zcwjk/policyDocumentLibrary)

本项目采用HuggingFace的QWen3-embedding-0.6B的embeeding模型作为语义构建，Postgresql作为embeeding向量数据库，Lora模型微调采用sentence-transformers框架

LLM API接口采用OpenAI包，对应接口配置以及密钥需要自行购买配置。

## 环境要求

- Python 3.12 或更高版本 (推荐3.14)

## 安装与启动

1. **克隆项目**

```bash
git clone https://github.com/NarKa-Yu/rag-ai-system.git
cd rag-ai-system
```

2. ##配置.env环境变量##

```bash
HF_TOKEN=huggingface的token，配置后可让embedding模型初次下载更快
PG_CONNECT=postgresql的链接字符串，例如postgresql://用户名:密码@localhost:端口/数据库
MODEL_NAME=OpenAI接口配置
API_KEY=OpenAI接口配置
BASE_URL=OpenAI接口配置
```

3. **运行项目**

```bash
python chat.py
```

4. **项目可交互界面**

```bash
http://127.0.0.1:8000
```
