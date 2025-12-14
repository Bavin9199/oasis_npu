import asyncio
import os
import re
import logging
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
model_type = "google/gemini-3-pro-preview"

agent_desc_path = "E:\\NPU\\P0\\OASIS\\oasis_npu\\character profile\\user_descriptions.csv"
agent_desc = pd.read_csv(agent_desc_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),  # 指定 UTF-8
        logging.StreamHandler()  # 同时输出到终端
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 屏蔽 openai / openrouter 的 info 日志
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openrouter").setLevel(logging.WARNING)

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
        dynamic_simulate_points = """
            {
                "维度": "输出内容中的标点符号特征需要动态匹配以下的特征进行拟真，在同一次输出中应该保持标点符号特征的一致性",
                "特征": [
                    "- 两个短句之间偶尔使用空格或不加标点直接连接，例如："今天真开心 明天也要努力"",
                    "- 标点可以重复，如 ",,,", "!!!", "??"",
                    "- 单个叹号，句号，省略号等标点符号在日常社交平台中并不常见，需要尽量避免使用",
                    "- 正常使用标点符号"
                ]
            },
            {
                "维度": "输出内容需要严格匹配对应的一种或几种字数特征，字数特征应该符合社交媒体的特点，内容不应太多",
                "特征": [
                    "- 普通网民评论 必须严格在 5–25 字，可搭配少量表情符号，微型影响者**必须严格在 15–30 字**；专业人士评论**必须严格在 10–40 字**；KOL/网红评论**不得超过 40 字**，确保内容完整。",
                    "- 开心情绪下评论**严格 3–15 字**，可重复“哈哈哈”“!!!”；生气或吐槽评论**严格 5–30 字**，符号密集；悲伤或思考评论**严格 10–40 字**，保持叙事或完整表达。",
                    "- 根据内容类型，观点表达或建议**严格 10–40 字**；吐槽或抱怨**严格 8–30 字**，可混合短句和长句；提问或互动**严格 3–20 字**；纯表情或 emoji 回复**严格 2–10 字**。",
                    ]
                Example:
                    - 评论长度必须严格在 3–15 字之间：
                    正确示例：哈哈，好开心啊！
                    错误示例：哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈

            }
        """
    
        critique_points = """
            {
                "维度": "语言是否过于正式、生硬、书面化",
                "含义": "AI 容易写成论文式、客服式、说明文式语气，而不是自然口语。",
                "症状": [
                    "句子太完整，没有口语停顿",
                    "用词偏书面，如‘然而’、‘此外’、‘因此’、‘值得注意的是’",
                    "缺少口语习惯用语，如‘我靠、呃、其实、感觉吧、说实话’"
                ],
                "改进建议": [
                    "加入口语表达，如‘说真的…’、‘我觉得…’",
                    "允许句子不完整、有停顿和转折",
                    "减少书面连接词，改成口语化‘然后呢…’、‘结果就…’"
                ]
            },
            {
                "维度": "逻辑是否过度完整、线性、像机械推演",
                "含义": "AI 文本通常逻辑过于完整、线性，而真实对话会跳跃、带情绪。",
                "症状": [
                    "段落明显按结构排列，像写作文",
                    "每句话都按‘原因→结论→补充’模式",
                    "没有思路中断、犹豫或临时转向"
                ],
                "改进建议": [
                    "加入自然思维跳跃，如‘欸不对，我想了想…’",
                    "允许逻辑不完整，如‘我也不太确定，但…’",
                    "加入临时自我纠正或犹豫表达"
                ]
            },
            {
                "维度": "表达是否不够自然、缺乏人类语气变化",
                "含义": "AI 语气常统一，缺少自然高低起伏。",
                "症状": [
                    "缺乏感叹、犹豫、吐槽",
                    "全篇语气一致，要么太开心，要么太中性",
                    "缺少自然语调词，如‘欸、哎呀、等会儿、算了’"
                ],
                "改进建议": [
                    "加入语气词、口头禅、反应词",
                    "允许情绪高低变化，比如突然兴奋或泄气",
                    "打破平直语调，如‘我当时整个人都愣住了’"
                ]
            },
            {
                "维度": "是否存在过多的赞同、迎合、缺乏独立判断",
                "含义": "AI 过度认可对方观点，无法根据自己特征提出新的观点。",
                "症状": [
                    "动不动说‘你说得很对’",
                    "无论对方说什么都附和",
                    "没有自己的判断，只是‘你觉得怎样就怎样’"
                ],
                "改进建议": [
                    "减少自动附和，适当表达不同观点",
                    "加入真实对话风格，如‘我不完全同意，但…’",
                    "保留独立立场，不盲目迎合"
                ]
            },
            {
                "维度": "是否输出表情包来凑‘人味’但显得刻意",
                "含义": "AI 容易用大量 emoji 填充情绪，像在模拟人却太用力。",
                "症状": [
                    "每句都加表情符号（尤其是😊😂✨🔥）",
                    "表情风格极度统一，而非自然",
                    "表情数量和情绪不匹配（严肃话题却😂）"
                ],
                "改进建议": [
                    "减少 emoji 密度，让它成为情绪点缀而非模板填充",
                    "让表情风格随语境变化，不固定使用",
                    "适当用文字代替表情（如‘我当时笑疯了’比😂更自然）"
                ]
            },
            {
                "维度": "是否强行使用比喻或类比",
                "含义": "AI 为了增加趣味或生动性，容易强行加比喻或类比，导致表达不自然。",
                "症状": [
                    "随意加比喻但与语境不贴合",
                    "比喻过于复杂或牵强，读起来像填充内容",
                    "比喻重复或模板化，如‘就像…一样’频繁出现"
                ],
                "改进建议": [
                    "只在自然、贴切的场景下使用比喻",
                    "避免复杂或牵强的类比，保持语言真实",
                    "比喻应增强理解或情感，而不是为了‘趣味’而加"
                ]
            },
            {
                "维度": "内容开头和结尾方式多样化",
                "含义": "AI 输出常习惯用提问结尾，习惯用哈哈同意等开头，缺少多样化的结束方式。",
                "症状": [
                    "每条内容结尾都以问题结束",
                    "缺少分享感受、总结观点或轻松收尾的方式",
                    "结尾单调，容易让对话显得机械或模板化"
                ],
                "改进建议": [
                    "可以用分享个人感受、总结观点、感叹或轻松收尾",
                    "结尾方式应根据语境和内容灵活选择",
                    "避免每次都用问题收尾，让对话更自然"
                ]
            }
        """

        prompt = f"""
            You are a social media user named {self.name}.
            Your profile is: {agent_desc}
            请从以下维度严格学习如何判断这段回复是否存在“AI 味”并避免出现“AI 味”的情况：
            {critique_points}
            请务必严格、全面，不要出现任何疑似 AI 生成的迹象。

            根据以下拟真规则，自动选择与该用户的人设特征（年龄、个性、情绪倾向、写作习惯及社交媒体风格）相符的规则，你必须严格遵守所选择的字数范围要求。  
            {dynamic_simulate_points}

            You have just received the following message:
            "{message}"

            Task:
            Reply like a real social media user in Chinese, based on your persona and the realism rules.

            Requirements:
            - Write one natural Chinese reply.
            - No explanations, no analysis, no AI tone.
            - Allow human-like imperfections: irregular punctuation, repeated symbols, casual spacing, varied length based on emotion/persona.
            - Sound spontaneous and realistic.
        """
        # ==========================
        # ---- 调用模型 ----
        # ==========================
        print("--------------")
        #print(prompt)
        response = client.chat.completions.create(
            model=model_type,
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
        #print("Selected response is:", model_output)
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
async def two_agent_chat(agent_a, agent_b, rounds=10):
    # 初始化对话
    last_msg = "Alice :动物物种灭绝（例如恐龙、渡渡鸟等）是自然的过程。有人认为人们不应该阻止这一自然过程。"
    logging.info(last_msg)
    #print(f"{agent_a.name}: {last_msg}")

    for i in range(rounds):
        logging.info(f"\n===== Round {i+1} =====")
        # --------------------------
        # ---- Agent B 回复 ----
        # --------------------------
        reply_b = await agent_b.respond(last_msg, agent_b.dynamic_desc)
        logging.info("BOB:" + reply_b)
        #print(f"{agent_b.name}: {reply_b}")
        last_msg = last_msg + "BOB:" + reply_b
        # --------------------------
        # ---- Agent A 回复 ----
        # --------------------------
        reply_a = await agent_a.respond(reply_b, agent_a.dynamic_desc)
        logging.info("Alice:" + reply_a)
        #print(f"{agent_a.name}: {reply_a}")

        last_msg = last_msg + "Alice:" + reply_a  # 下一轮传给 Agent B
    
    print(last_msg)
    #print("\n")

# ==========================================================

if __name__ == "__main__":
    import asyncio

    # ---- 创建两个 Agent ----
    agent_a = SimpleAgent(id=0, name="Alice", dynamic_desc=get_openai_messages(0))
    agent_b = SimpleAgent(id=1, name="Bob", dynamic_desc=get_openai_messages(1))


    # ---- 运行双 Agent 对话 ----
    try:
        asyncio.run(two_agent_chat(agent_a, agent_b, rounds=10))
    except Exception as e:
        print("[ERROR]", e)