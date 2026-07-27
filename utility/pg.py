import psycopg
import os
from typing import Callable

conn_str = os.getenv('PG_CONNECT')
conn_str = 'postgresql://postgres:narka@localhost:5432/postgres'

def create_table1():
    '''创建documents表 用于存放文档解析后的chunk信息'''

    create_table_sql = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documents (
        id          SERIAL PRIMARY KEY,
        title       VARCHAR(200) DEFAULT '',
        chunk       VARCHAR(800) DEFAULT NULL,
        embed       VECTOR(1024) DEFAULT NULL
    );

    -- 创建索引 加速向量检索
    CREATE INDEX IF NOT EXISTS idx_documents_embedding 
    ON documents 
    USING hnsw (embed vector_cosine_ops);
    """
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            conn.commit()
            print("表格创建成功")

def create_table2():
    '''创建querys表 用于存放构建训练样本的querys'''

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS querys (
        id          SERIAL PRIMARY KEY,
        query       VARCHAR(500) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_querys_query ON querys (query);
    """
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            conn.commit()
            print("querys表格创建成功")

def create_table3():
    '''创建samples表 用于存放训练embedding模型的[query, chunk, label]样本'''

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS samples (
        id          SERIAL PRIMARY KEY,
        query_id    INT NOT NULL,
        chunk_id    INT NOT NULL,
        label       INT NOT NULL CHECK (label IN (0, 1))
    );
    """
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            conn.commit()
            print("samples表格创建成功")


def insert_chunks(chunks: list[(str, str, list[float])]):
    '''把title chunk embedding形式的记录存入数据库中'''
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            for title, chunk, embed in chunks:
                cur.execute("""INSERT INTO documents (title, chunk, embed)
                VALUES (%s, %s, %s::vector);""", (title, chunk, embed))
                conn.commit()


def get_similar_chunks(embedding: list[float], top_k: int = 5):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT id, title, chunk, 1 - (embed <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embed <=> %s::vector ASC
            LIMIT %s;
            """, (embedding, embedding, top_k))
            return cur.fetchall()

def make_samples(input: dict):
    query: str = input.get('query', None)
    samples: list[dict] = input.get('samples', [])

    if len(samples) == 0:
        return
    
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # WITH的作用是在sql查询中临时定义几个子查询 后续可以像普通表一样引用这些临时表
            # 查看querys表中有没有当前query 如果没有则创建一条query
            cur.execute("""
                WITH existing AS (
                    SELECT id 
                    FROM querys 
                    WHERE query = %s
                ),
                inserted AS (
                    INSERT INTO querys (query)
                    SELECT %s
                    WHERE NOT EXISTS (SELECT 1 FROM existing)
                    RETURNING id
                )
                SELECT id FROM existing
                UNION ALL
                SELECT id FROM inserted
                LIMIT 1;
            """, (query, query))
            
            result = cur.fetchone()
            query_id = result[0] if result else None

            for sample in samples:
                chunk_id = sample.get('id', None)
                label = sample.get('label', 0)
                cur.execute("""
                    INSERT INTO samples (query_id, chunk_id, label)
                    SELECT %s, %s, %s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM samples 
                        WHERE query_id = %s AND chunk_id = %s
                    );""", (query_id, chunk_id, label, query_id, chunk_id))

def clear_samples():
    '''清除samples表'''
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute('truncate table samples;')

def get_training_samples():
    '''获取训练集query和chunk都是字符串 label为1表示正例 label为0表示负例'''
    sql = """
        SELECT s.id, q.query, d.chunk, s.label
        FROM samples s
        JOIN querys q ON s.query_id = q.id
        JOIN documents d ON s.chunk_id = d.id
        ORDER BY s.id;
    """
    samples = []
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            for row in rows:
                id, query, chunk, label = row
                samples.append([id, query, chunk, label])
    return samples

def reset_embedding(embed_func: Callable[[str], list[float]]):
    '''重置documents表每个chunk的embed'''
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("select id, chunk from documents;")
            rows = cur.fetchall()
            for id_, chunk_ in rows:
                emb = embed_func(chunk_)
                cur.execute('UPDATE documents SET embed = %s::vector WHERE id = %s', (emb, id_))
                




if __name__ == "__main__":
    reset_embedding()
    # create_table3()
    # print(f"get_training_samples() = {get_training_samples()}")