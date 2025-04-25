import datetime
import os
import uuid

import duckdb
import litellm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"


DB_NAME = "dialogue.db"
TABLE_NAME = "summaries"
CHAT_MODEL_NAME = "ollama/hf.co/mmnga/ArrowMint-Gemma3-4B-YUKI-v0.1-gguf:latest"
SUMMARY_MODEL_NAME = "ollama/gemma3:27b"
EMBEDDING_MODEL_NAME = "pfnet/plamo-embedding-1b"
EMBEDDING_DIM = 2048


class ChatAgent:
    def __init__(self, name: str, model):
        self.name = name
        self.model = model

        self.system_prompt = "あなたは会話のスペシャリストです．今から自然な会話を行います．返答は短く口語で行ってください．言語は日本語です．"

    def chat(self, conversation: list[dict[str, str]]) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ] + [
            {
                "role": "assistant" if turn["name"] == self.name else "user",
                "content": turn["message"],
            }
            for turn in conversation
        ]
        response = litellm.completion(model=self.model, messages=messages)

        return response["choices"][0]["message"]["content"]  # type: ignore


def generate_chat_theme() -> str:
    response = litellm.completion(
        model=CHAT_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "あなたは会話のスペシャリストです．今から自然な会話を行います．返答は短く口語で行ってください．言語は日本語です．",
            },
            {
                "role": "user",
                "content": "2人のエージェント間の会話のテーマを提案してください。自由な発想を行って，ユニークで奇抜なテーマを考えてください．フォーマットに従い、それ以外の内容は返答しないでください。：## Theme: <テーマ>",
            },
        ],
        temperature=2.0,
    )
    chat_theme = response["choices"][0]["message"]["content"].split("## Theme: ")[1]  # type: ignore
    return chat_theme


def generate_conversation_summary(conversation: list[dict[str, str]]) -> str:
    response = litellm.completion(
        model=SUMMARY_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "あなたは会話のスペシャリストです．ユーザーが提供した会話の内容を要約してください．言語は日本語です．以下のフォーマットに従ってください．## Summary: <要約>",
            },
            {
                "role": "user",
                "content": f"以下の会話を要約してください：{conversation}",
            },
        ],
    )
    content = response["choices"][0]["message"]["content"]  # type: ignore
    try:
        summary = content.split("## Summary: ")[1]
    except IndexError:
        print("Error: Unable to extract summary from response.")
        summary = content
    return summary


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        con.sql(f"SELECT * FROM {table_name} LIMIT 1")
        return True
    except duckdb.CatalogException:
        return False


class PlamoEmbedding:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.device = "cpu"
        self.model = self.model.to(self.device)

    def encode_query(self, query: str) -> list[float]:
        return (
            self.model.encode_query(query, self.tokenizer)[0].detach().numpy().tolist()
        )

    def encode_document(self, document: str) -> list[float]:
        return (
            self.model.encode_document(document, self.tokenizer)[0]
            .detach()
            .numpy()
            .tolist()
        )


def main():
    embedding = PlamoEmbedding(EMBEDDING_MODEL_NAME)

    db_path = f"data/{DB_NAME}"
    con = duckdb.connect(db_path)

    if not table_exists(con, TABLE_NAME):
        print(f"Creating table {TABLE_NAME}...")
        con.sql(
            f"""
            CREATE TABLE {TABLE_NAME} (asctime TIMESTAMP, id UUID, content TEXT, embedding FLOAT[{EMBEDDING_DIM}])
            """
        )
        print(f"Table {TABLE_NAME} created.")

    latest_summaries = con.sql(
        f"""
        SELECT asctime, content FROM {TABLE_NAME} ORDER BY asctime DESC LIMIT 3
        """
    )
    print(latest_summaries)

    num_turns = 3
    conversation = []

    chat_theme = generate_chat_theme()
    print(f"Chat theme: {chat_theme}")

    agent_A = ChatAgent(name="A", model=CHAT_MODEL_NAME)
    agent_B = ChatAgent(name="B", model=CHAT_MODEL_NAME)

    for agent in [agent_A, agent_B]:
        agent.system_prompt = (
            agent.system_prompt
            + f"テーマは「{chat_theme}」です．会話のテーマに沿った内容で会話を行ってください．"
        )

    for turn in range(1, num_turns + 1):
        print(f"Turn {turn}:")

        response = agent_A.chat(conversation)
        conversation.append({"name": "A", "message": response})
        print(f"A: {response}")

        response = agent_B.chat(conversation)
        conversation.append({"name": "B", "message": response})
        print(f"B: {response}")
    print("Conversation finished.")

    summary = generate_conversation_summary(conversation).strip()
    print(f"Conversation summary:\n {summary}")

    summary_embedding = embedding.encode_document(summary)

    try:
        con.execute(
            f"""
            INSERT INTO {TABLE_NAME} (asctime, id, content, embedding)
            VALUES (?, ?, ?, ?)
            """,
            [datetime.datetime.now(), uuid.uuid4(), summary, summary_embedding],
        )

    except Exception as e:
        print(f"Error inserting data into {TABLE_NAME}: {e}")

    query_embedding = embedding.encode_query(chat_theme)
    sql_query = f"""
    SELECT *,
    array_cosine_similarity(embedding, {query_embedding}::FLOAT[{EMBEDDING_DIM}]) AS score
    FROM {TABLE_NAME}
    ORDER BY score DESC
    LIMIT 3
    """
    result = con.sql(sql_query)
    print(result)

    con.close()
    print("Database connection closed.")


if __name__ == "__main__":
    main()
