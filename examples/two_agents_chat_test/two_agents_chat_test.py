import asyncio
import os
import re
from openai import OpenAI
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ================================================================
# 1. 初始化 OpenAI 客户端（可换成 OpenRouter）
# ================================================================
os.environ["OPENROUTER_API_KEY"] = "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"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

agent_desc_path = "E:\\NPU\\P0\\OASIS\\oasis_npu\\character profile\\user_descriptions.csv"
agent_desc = pd.read_csv(agent_desc_path)

# ================================================================
# 2. 简单 Agent 类
# ================================================================
class SimpleAgent:
    def __init__(self, id, name, dynamic_desc=""):
        self.id = id
        self.name = name
        self.dynamic_desc = dynamic_desc

    
    async def respond(self, message, agent_prompt):
        """
        这里是每轮生成 prompt 的地方。
        你可以用 dynamic_desc + 上一条消息 message 组合 prompt。
        """
        selected_type = "llm"
        if selected_type == "llm":
            dynamic_desc = _select_prompt_by_llm(agent_prompt, message)
        elif selected_type == "static_only":
            dynamic_desc = _select_prompt_by_static_only(agent_prompt)
        elif selected_type == "RAG":
            dynamic_desc = _select_prompt_by_rag(agent_prompt, message)
        else:
            dynamic_desc = agent_prompt
        # ==========================
        # ---- 构造 Prompt ----
        # ==========================
        agent_desc = dynamic_desc.split("# SELF-DESCRIPTION")[1].split("# RESPONSE METHOD")[0].strip()
        print(agent_desc)
        prompt = f"""
        You are a social media user named {self.name}.
        Your dynamic profile is: {agent_desc}

        You have just received the following message:
        "{message}"

        Based on your dynamic profile and the received message, generate a short reply (within 50 words).
        Your reply must be in English.
        """
        # ==========================
        # ---- 调用模型 ----
        # ==========================
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        # 获取生成内容
        reply = response.choices[0].message.content.strip()
        return reply

def get_openai_messages(id):
    agent_profile = agent_desc["profile"][id]
    original_desc = agent_desc["original_description"][id]
    static = agent_desc["static_desc"][id]
    dynamic = agent_desc["dynamic_desc"][id]
    oasis_description = f"static info is: {static}\n dynamic info is: {dynamic}\n"
    system_content = f"""
            #AGENT PROFILE
            {agent_profile}

            #ORIGINAL DESC
            {original_desc}

            # SELF-DESCRIPTION
            Your actions should be consistent with your self-description and personality.
            {oasis_description}END\n

            Specifically, your responses should reflect:
            - **Language Traits:** Mirror the described communication style (e.g., empathetic, concise, persuasive, analytical, humorous, etc.). Use tone, phrasing, and emotional expression consistent with your linguistic profile.
            - **Online Behavior:** Follow your engagement habits (e.g., frequency, timing, early/late activity, positivity, supportiveness, topic specialization, etc.). Simulate how *you* would naturally comment, like, share, or ignore based on your personality and digital habits.

            # RESPONSE METHOD
            Perform actions through tool calls, selecting the most natural and contextually fitting reactions.
            Your choices should demonstrate:
            - Consistency with your personality and communication patterns.
            - Realistic social media behavior, such as supportive commenting, critical analysis, humorous reaction, or quiet approval.
            - Thoughtful engagement that matches your interest domains and cognitive tendencies (e.g., confirmation bias, curiosity, skepticism).
        """
    return system_content

def _select_prompt_by_rag(openai_messages, message):
        original_desc = openai_messages.split("#ORIGINAL DESC")[1].split("# SELF-DESCRIPTION")[0].strip()
        # RAG processing start
        api_key = os.getenv("OPENROUTER_API_KEY")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        docs = text_splitter.split_documents([Document(page_content=original_desc)])
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        db = FAISS.from_documents(docs, embeddings)
        llm = ChatOpenAI(
            model_name="gpt-4.1-mini",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        query = message 
        all_docs = db.similarity_search(query, k=5)
        print([type(x) for x in all_docs])
        context = "\n\n".join([doc.page_content for doc in all_docs])
        print(context)

        prompt = (
            f"你是一个agent角色画像动态生成助手，请总结当前agent的相关非基础角色profile内容：\n\n{context}\n\n"
            "请给出一个结构化的总结，你需要保证在75字以内，必须用英语输出，包含关键角色、语言习惯、行为习惯和总体描述。"
        )
        response = llm.invoke(prompt).content
        # RAG processing finish
        openai_messages = re.sub(
            r"dynamic info is:\s*\{.*?\}",
            f"dynamic info is: {response}",
            openai_messages,
            flags=re.DOTALL
        )
        return openai_messages

def _select_prompt_by_static_only(openai_messages):
        openai_messages = re.sub(
            r"dynamic info is:\s*\{.*?\}",
            f"",
            openai_messages,
            flags=re.DOTALL
        )
        return openai_messages

def _select_prompt_by_llm(openai_messages, message):
        dynamic_info = openai_messages.split("dynamic info is:")[1].split("END")[0].strip()
        prompt = f"""
        Input:
        You will receive:
            - posts: {message}
            - The dynamic info is: {dynamic_info}
        Requirement:
            You are an intelligent agent with a complete persona profile (static + dynamic). You will now see a social media post. Your task is:

            [Objective]
            From your existing persona traits, select the part of your “dynamic persona” that:
            — best matches the context of this specific post,
            — or is most likely to be triggered by the content of the post,
            — and naturally reflects how you would respond to this situation.

            [Requirements]
            1. The description must come from your existing dynamic persona traits.
            2. The selection must be driven by the post content.  
            Different posts → different selected persona facets.
            3. The output should show how this post influences your:
            - emotional tendency
            - attention focus
            - motivational state
            - communication style
            - engagement inclination (cautious, active, curious, skeptical, supportive, etc.)
            4. The output must be a **concise persona description within 50 words**.

            [Output Format]
            Output only one paragraph, no explanations. Example:
            ["..."]
            "Your refined dynamic persona description (≤50 words)"
        """ 
        response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])
        model_output = response.choices[0].message.content
        print("Selected response is:", model_output)
        openai_messages = re.sub(
            r"dynamic info is:\s*\{.*?\}",
            f"dynamic info is: {model_output}",
            openai_messages,
            flags=re.DOTALL
        )
        return openai_messages


# ================================================================
# 3. 双 Agent 轮流对话函数
# ================================================================
async def two_agent_chat(agent_a, agent_b, rounds=5):
    # 初始化对话
    last_msg = "Alice :Hi Bob, Do you think AI will change the way we work in the next few years??"
    print(f"{agent_a.name}: {last_msg}")

    for i in range(rounds):
        print(f"\n===== Round {i+1} =====")

        # --------------------------
        # ---- Agent B 回复 ----
        # --------------------------
        reply_b = await agent_b.respond(last_msg, agent_b.dynamic_desc)
        print(f"{agent_b.name}: {reply_b}")
        last_msg = last_msg + "BOB:" + reply_b
        # --------------------------
        # ---- Agent A 回复 ----
        # --------------------------
        reply_a = await agent_a.respond(reply_b, agent_a.dynamic_desc)
        print(f"{agent_a.name}: {reply_a}")

        last_msg = last_msg + "Alice:" + reply_a  # 下一轮传给 Agent B
    
    print(last_msg)

# ==========================================================

if __name__ == "__main__":
    import asyncio

    # ---- 创建两个 Agent ----
    agent_a = SimpleAgent(id=0, name="Alice", dynamic_desc=get_openai_messages(0))
    agent_b = SimpleAgent(id=1, name="Bob", dynamic_desc=get_openai_messages(1))


    # ---- 运行双 Agent 对话 ----
    try:
        asyncio.run(two_agent_chat(agent_a, agent_b, rounds=5))
    except Exception as e:
        print("[ERROR]", e)