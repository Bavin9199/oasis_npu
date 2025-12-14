"""
llm_describer.py

Requirements:
  pip install openai tenacity

Usage:
  export OPENAI_API_KEY="sk-..."
  python -m llm_describer  # runs demo at bottom
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from random_username.generate import generate_username
from datetime import datetime
import os
os.environ["OPENROUTER_API_KEY"] = "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"
import json
import random
from typing import Dict, List, Any
import textwrap
import time

from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"])

# --------------------------
# Basic Info: set and extract
# --------------------------

# Gender ratio
gender_ratio = [0.351, 0.636]
genders = ['female', 'male']

# Age ratio
age_ratio = [0.44, 0.31, 0.11, 0.03, 0.11]
age_groups = ['18-29', '30-49', '50-64', '65-100', 'underage']

# MBTI ratio
p_mbti = [
    0.12625, 0.11625, 0.02125, 0.03125, 0.05125, 0.07125, 0.04625, 0.04125,
    0.04625, 0.06625, 0.07125, 0.03625, 0.10125, 0.11125, 0.03125, 0.03125
]
mbti_types = [
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP",
    "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

# Country ratio
country_ratio = [0.4833, 0.0733, 0.0697, 0.0416, 0.0306, 0.3016]
countries = ["US", "UK", "Canada", "Australia", "Germany", "Other"]

# Profession ratio
p_professions = [1 / 16] * 16
professions = [
    "Agriculture, Food & Natural Resources", "Architecture & Construction",
    "Arts, Audio/Video Technology & Communications",
    "Business Management & Administration", "Education & Training", "Finance",
    "Government & Public Administration", "Health Science",
    "Hospitality & Tourism", "Human Services", "Information Technology",
    "Law, Public Safety, Corrections & Security", "Manufacturing", "Marketing",
    "Science, Technology, Engineering & Mathematics",
    "Transportation, Distribution & Logistics"
]

def get_random_gender():
    return random.choices(genders, gender_ratio)[0]

def get_random_age():
    group = random.choices(age_groups, age_ratio)[0]
    if group == 'underage':
        return random.randint(10, 17)
    elif group == '18-29':
        return random.randint(18, 29)
    elif group == '30-49':
        return random.randint(30, 49)
    elif group == '50-64':
        return random.randint(50, 64)
    else:
        return random.randint(65, 100)

def get_random_mbti():
    return random.choices(mbti_types, p_mbti)[0]

def get_random_country():
    country = random.choices(countries, country_ratio)[0]
    if country == "Other":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "Select a real country name randomly:"
            }])
        return response.choices[0].message.content.strip()
    return country

def get_random_profession():
    return random.choices(professions, p_professions)[0]

def get_interested_topics(mbti, age, gender, country, profession):
    prompt = f"""Based on the provided personality traits, age, gender and profession, please select 1-2 topics of interest from the given list.
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Country: {country}
        Profession: {profession}
    Available Topics:
        1. Economics: The study and management of production, distribution, and consumption of goods and services. Economics focuses on how individuals, businesses, governments, and nations make choices about allocating resources to satisfy their wants and needs, and tries to determine how these groups should organize and coordinate efforts to achieve maximum output.
        2. IT (Information Technology): The use of computers, networking, and other physical devices, infrastructure, and processes to create, process, store, secure, and exchange all forms of electronic data. IT is commonly used within the context of business operations as opposed to personal or entertainment technologies.
        3. Culture & Society: The way of life for an entire society, including codes of manners, dress, language, religion, rituals, norms of behavior, and systems of belief. This topic explores how cultural expressions and societal structures influence human behavior, relationships, and social norms.
        4. General News: A broad category that includes current events, happenings, and trends across a wide range of areas such as politics, business, science, technology, and entertainment. General news provides a comprehensive overview of the latest developments affecting the world at large.
        5. Politics: The activities associated with the governance of a country or other area, especially the debate or conflict among individuals or parties having or hoping to achieve power. Politics is often a battle over control of resources, policy decisions, and the direction of societal norms.
        6. Business: The practice of making one's living through commerce, trade, or services. This topic encompasses the entrepreneurial, managerial, and administrative processes involved in starting, managing, and growing a business entity.
        7. Fun: Activities or ideas that are light-hearted or amusing. This topic covers a wide range of entertainment choices and leisure activities that bring joy, laughter, and enjoyment to individuals and groups.
    Output:
    [string of topics]
    ["artificial intelligence", "technology trends"]
    Ensure your output looks like the output above, don't output anything else.""" 

    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)

def get_social_roles(mbti, age, gender, country, profession):
    social_roles_content = load_json("social_roles.json")
    prompt = f"""Based on the provided personality traits, age, gender and profession, please select 1 social role from the given list.-
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Country: {country}
        Profession: {profession}
    Available Social Roles and related descriptions:
        {social_roles_content}
    Output:
    [string of topics]
    ["artificial intelligence"]
    Ensure your output looks like the output above, don't output anything else.""" 

    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)

def get_cognitive_bias(mbti, age, gender, country, profession):
    cognitive_bias_content = load_json("cognitive_bias.json")
    prompt = f"""Based on the provided personality traits, age, gender and profession, please select 2-3 cognitive bias from the given list.-
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Country: {country}
        Profession: {profession}
    Available Cognitive bias and related descriptions:
        {cognitive_bias_content}
    Output:
    [string of topics]
    ["artificial intelligence", "technology trends", "psychology"]
    Ensure your output looks like the output above, don't output anything else.""" 

    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)

def get_network_levels(mbti, age, gender, country, profession, social_roles, cognitive_bias):
    network_level_content = load_json("network_level.json")
    prompt = f"""Based on the provided personality traits, age, gender, profession, social roles and cognitive bias, please select 1 network level from the given json list.-
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Country: {country}
        Profession: {profession}
        Social roles: {social_roles}
        Cognitive bias: {cognitive_bias}
    Available Network Levels are : Ordinary User, Micro-influencer and Social Media Celebrity (Macro-influencer).
    and related descriptions:
        {network_level_content}
    Output:
    [string of topics]
    ["..."]
    Ensure your output looks like the output above in one line, don't output anything else.""" 

    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)

def get_action_habits(mbti, age, gender, country, profession, social_roles, cognitive_bias, network_levels):
    action_habits_content = load_json("action_habit.json")
    prompt = f"""Based on the provided personality traits, age, gender, profession, social roles and cognitive bias, please select 2-3 action habits from the given list.-
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Country: {country}
        Profession: {profession}
        Social roles: {social_roles}
        Cognitive bias: {cognitive_bias}
        network_levels: {network_levels}
    Available Action Habits and related descriptions:
        {action_habits_content}
    Please choose 2-3 action habits from the second_level topic in available action habits list based on the profile information carefully.
    Output:
    [string of topics]
    ["artificial intelligence"]
    Ensure your output looks like the output above, don't output anything else.""" 
    response = client.chat.completions.create(model="gpt-4o-mini",
                                            messages=[{
                                                "role": "system",
                                                "content": prompt
                                            }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)
    
def get_user_name(mbti, age, gender, country, profession, social_roles, cognitive_bias, network_levels):
    username_list = generate_username(20)
    prompt = f"""Based on the provided personality traits, age, gender, profession, social roles and cognitive bias, please select one proper user name from the username list.-
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Country: {country}
        Profession: {profession}
        Social roles: {social_roles}
        Cognitive bias: {cognitive_bias}
        network_levels: {network_levels}
        user_name_list: {username_list}
    Output:
    [string of topics]
    ["XXX"]
    Ensure your output ONLY contains one username, don't output anything else.""" 

    response = client.chat.completions.create(model="gpt-4o-mini",
                                            messages=[{
                                                "role": "system",
                                                "content": prompt
                                            }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)

# --------------------------
# Utilities: load and extract
# --------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_texts_from_entry(entry: Any) -> List[str]:
    """
    Recursively extract all string values from an entry (dict/list/str).
    Returns list of cleaned strings.
    """
    texts = []
    if isinstance(entry, str):
        s = entry.strip()
        if s:
            texts.append(s)
    elif isinstance(entry, dict):
        for v in entry.values():
            texts.extend(extract_texts_from_entry(v))
    elif isinstance(entry, list):
        for v in entry:
            texts.extend(extract_texts_from_entry(v))
    return texts

def summarize_entries_for_keys(data_dict: Dict[str, Any], keys: List[str]) -> Dict[str, str]:
    """
    For a dict of entries (like action_habit), and a list of keys to look up,
    return a mapping {key: joined_text}.
    If an entry has many text fields, they will be concatenated with proper spacing.
    """
    out = {}
    for k in keys:
        if isinstance(data_dict, list):
            # 如果是列表，尝试遍历其中的字典对象寻找对应 key
            entry = next((item for item in data_dict if isinstance(item, dict) and k in item), None)
        else:
            entry = data_dict.get(k)
        if entry is None:
            continue
        texts = extract_texts_from_entry(entry)
        if texts:
            out[k] = " ".join(texts)
    return out

def build_user_profile():
    mbti = get_random_mbti()
    age = get_random_age()
    gender = get_random_gender()
    country = get_random_country()
    profession = get_random_profession()
    interested_topics = get_interested_topics(mbti, age, gender, country, profession)
    social_roles = get_social_roles(mbti, age, gender, country, profession)
    cognitive_bias = get_cognitive_bias(mbti, age, gender, country, profession)
    network_levels = get_network_levels(mbti, age, gender, country, profession, social_roles, cognitive_bias)
    action_habits = get_action_habits(mbti, age, gender, country, profession, social_roles, cognitive_bias, network_levels)
    user_name = get_user_name(mbti, age, gender, country, profession, social_roles, cognitive_bias, network_levels)
    
    profile = {}
    profile["user_name"] = user_name
    profile['age'] = age
    profile['gender'] = gender
    profile['mbti'] = mbti
    profile['country'] = country
    profile['profession'] = profession
    profile['interested_topics'] = interested_topics
    profile['social_roles'] = social_roles
    profile['cognitive_bias'] = cognitive_bias
    profile['network_level'] = network_levels
    profile['action_habit'] = action_habits
    return profile

# --------------------------
# Prompt builder
# --------------------------
def build_prompt(profile: Dict[str, Any],
                 datasets: Dict[str, Dict[str, Any]],
                 *,
                 language: str = "English",
                 tone: str = "neutral",
                 max_paragraphs: int = 3,
                 max_length: int = 220) -> str:
    """
    Construct a prompt to instruct the LLM to write a compact, high-quality profile description.
    `profile` may contain lists per key.
    `datasets` must contain keys mapping to loaded JSON dicts.
    """
    # Normalize input values to lists
    normalized = {}
    for k in datasets.keys():
        v = profile.get(k)
        if v is None:
            normalized[k] = []
        elif isinstance(v, list):
            normalized[k] = v
        else:
            normalized[k] = [v]

    # For each dimension, gather text snippets for chosen keys
    dimension_snippets = {}
    for dim, data in datasets.items():
        keys = normalized.get(dim, [])
        if not keys:
            dimension_snippets[dim] = []
            continue
        mapping = summarize_entries_for_keys(data, keys)
        snippets = [f"{k}: {mapping.get(k, '(no data)')}" for k in keys]
        dimension_snippets[dim] = snippets
    mbti = profile.get("mbti")
    age = profile.get("age")
    gender = profile.get("gender")
    profession = profile.get("profession")
    social_roles = profile.get("social_roles")
    cognitive_bias = profile.get("cognitive_bias")
    network_levels = profile.get("network_levels")
    action_habits = profile.get("action_habits")
    user_name = profile.get("user_name")

    # Compose the enhanced prompt
    prompt_lines = []
    prompt_lines.append(f"You are an expert profile writer that creates natural, cohesive, and factual persona summaries in {language}.")
    prompt_lines.append("You will receive structured personality and behavior attributes along with short reference texts for each attribute.")
    prompt_lines.append("Your task is to locate the corresponding information in the reference texts and integrate all relevant descriptions into a unified persona summary.")
    prompt_lines.append("Your task is to integrate them into a natural description paragraph, starting with the person's age, gender, and profession — not to list or copy the attributes.")
    prompt_lines.append("When multiple descriptions exist for the same attribute, merge the common points while preserving the unique details. Do not simply list or copy the text.")
    prompt_lines.append(f"Tone: {tone}. Length: up to {max_paragraphs} paragraph(s), each around {max_length} characters.")
    prompt_lines.append("Do not invent new facts, but you may rephrase or combine ideas to make the text flow naturally. Avoid repeating raw definitions.")
    prompt_lines.append("Write in full sentences that sound human and smooth, suitable for displaying in a profile or agent summary box.")
    prompt_lines.append("The static description is only about the information of name, gender, mbti, cognitive bias, professions and static personal things which couldn't be changed by the environments.")
    prompt_lines.append("After writing the main persona description, append two additional lines summarizing key patterns with related clear prefix:")
    prompt_lines.append("1) Language Traits: A line describing all notable characteristics related to the individual's language and communication style, such as tone, empathy, persuasiveness, listening habits, and inclusivity.")
    prompt_lines.append("2) Online Behavior: A line describing all notable patterns in the individual's behavior in digital or social networks, such as activity rhythms, schedule flexibility, social roles, network level, and biases in information consumption.")
    prompt_lines.append("Ensure these two summary lines are concise, factually grounded in the input attributes, and written in fluent, natural English.")
    prompt_lines.append("Input attributes and reference snippets:")
    prompt_lines.append(f"The user name of this person is {user_name}. The person is a {age}-year-old {gender} people. The mbti of this person is {mbti}. The profession of this person is {profession}. The social roles of this person is {social_roles}.")
    prompt_lines.append(f"The cognitive bias of this person is {cognitive_bias}. The network levels of this person is {network_levels}. The action_habits of this person is {action_habits}.")
    prompt_lines.append("----")
    prompt_lines.append("Output format:")
    prompt_lines.append("""{
        "static description": "...", 
        "dynamic description":""language_traits": "...", "online_behavior": "...""
    }""")
    prompt_lines.append("Output ONLY JSON, no extra text or explanation.")

    for dim, snippets in dimension_snippets.items():
        prompt_lines.append(f"{dim}:")
        if snippets:
            for s in snippets:
                wrapped = textwrap.fill(s, width=100, subsequent_indent="    ")
                prompt_lines.append(f"  - {wrapped}")
        else:
            prompt_lines.append("  - (none)")
        prompt_lines.append("")

    prompt_lines.append("----")
    prompt_lines.append("Task:")
    prompt_lines.append("1) Write one fluent, natural paragraph that synthesizes all relevant aspects of the person.")
    prompt_lines.append("2) Maintain factual grounding in the snippets, but express it cohesively.")
    prompt_lines.append("3) Do not include bullets, labels, or formatting — just a natural English paragraph.")
    prompt_text = "\n".join(prompt_lines)
    return prompt_text

# --------------------------
# LLM call (OpenRouter version)
# --------------------------
import requests
import os

def call_llm(prompt: str,
             *,
             model: str = "gpt-4o-mini",
             temperature: float = 0.7,
             max_tokens: int = 300) -> str:
    """
    Call OpenRouter chat completion API instead of OpenAI.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise, factual profile writer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

    result = resp.json()
    # 返回生成文本
    return result['choices'][0]['message']['content'].strip()


# --------------------------
# Main wrapper
# --------------------------
def generate_description(*, use_llm: bool = True) -> Dict[str, Any]:
    """
    profile: keys are the five dimensions; values can be str or list[str]
    data_paths: mapping dimension->json filename
    Returns: {"prompt":..., "description":..., "method": "llm" or "local_fallback"}
    """
    # load agent profile
    profile = build_user_profile() 

    # load datasets
    data_paths = {
        "action_habit": "action_habit.json",
        "cognitive_bias": "cognitive_bias.json",
        "mbti": "mbti.json",
        "network_level": "network_level.json",
        "social_roles": "social_roles.json"
    }
    datasets = {}
    for dim, path in data_paths.items():
        datasets[dim] = load_json(path)

    prompt = build_prompt(profile, datasets, language="English", tone="neutral")
    result = {}

    if use_llm:
        try:
            description = call_llm(prompt)
            result = {}
            try:
                json_response = json.loads(description)
                result["profile"] = profile
                result["dynamic_info"] = json_response.get("dynamic description", {})
                result["static_info"] = json_response.get("static description", {})
                fallback = local_fallback_summary(profile, datasets)
                result["original_description"] = fallback
            except json.JSONDecodeError:
                json_response = {"description": description}
            return result
        except Exception as e:
            # fall back to local concatenation
            result["llm_error"] = str(e)
            # continue to local fallback

    # Local fallback: collect texts and join into readable summary
    #parts = []
    #for dim, data in datasets.items():
    #    vals = profile.get(dim, [])
    #    if not isinstance(vals, list):
    #        vals = [vals]
    #    for v in vals:
    #        if isinstance(data, list):
    #            # 如果是列表，则在列表中查找包含该键的字典
    #            entry = next((item.get(v) for item in data if isinstance(item, dict) and v in item), None)
    #        else:
    #            entry = data.get(v)
    #        if entry:
    #            texts = extract_texts_from_entry(entry)
    #            if texts:
    #                parts.append(" ".join(texts))
    #fallback = " ".join(parts) if parts else "(no data available to describe user)"
    #result = fallback.split("'description':", 1)[-1].strip(" {}'\"")
    # result.update({"description": fallback})
    #return result

def local_fallback_summary(profile: Dict[str, Any],
                           datasets: Dict[str, Dict[str, Any]]) -> str:
    """
    Build a structured fallback description when LLM fails.
    Each dimension (corresponding to each JSON file) will have its own section.
    """
    parts = []

    for dim, data in datasets.items():           # dim = 文件名（action/bias/...）
        vals = profile.get(dim, [])
        print(vals)
        if not isinstance(vals, list):
            vals = [vals]

        dim_texts = []

        for v in vals:
            entry = None

            # JSON 如果是列表（如 [{key: {...}}, {}]）
            if isinstance(data, list):
                entry = next(
                    (item.get(v) for item in data if isinstance(item, dict) and v in item),
                    None
                )
            else:
                # JSON 是 dict 格式
                entry = data.get(v)

            if entry:
                texts = extract_texts_from_entry(entry)
                if texts:
                    dim_texts.extend(texts)

        # 当前文件有文本 → 输出标题 + 内容
        if dim_texts:
            section = f"[{dim}]" + " ".join(dim_texts)
            parts.append(section)

    # 每一块用空行隔开，更清晰
    return " ".join(parts) if parts else "(no data available to describe user)"

def generate_user_descriptions(n):
    # user_description = []
    start_time = datetime.now()
    max_workers = 10
    with open("user_descriptions.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "profile", "original_description", "static_desc", "dynamic_desc"])
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_description) for _ in range(n)]
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                profile = result["profile"]
                original_description = result["original_description"]
                static_desc = result["static_info"]
                dynamic_desc = result["dynamic_info"]
                writer.writerow([i, profile, original_description, static_desc, dynamic_desc])
                
                elapsed_time = datetime.now() - start_time
                print(f"Generated {i+1}/{n} user profiles. Time elapsed: "
                    f"{elapsed_time}")

if __name__ == "__main__":
    N = 3  # Target user number
    user_desc = generate_user_descriptions(N)
    print(f"Generated {N} user profiles and saved.")


# --------------------------
# Demo / CLI (example)
# --------------------------
# Example profile with multiple attributes per dimension
    example_profile = {
        "age": 25,
        "gender": "male",
        "job": "software engineer",
        "action_habit": ["night_owl", "polite_tone"],
        "cognitive_bias": ["negativity_bias"],
        "mbti": ["INFJ"],
        "network_level": ["micro_influencer"],
        "social_roles": ["Engineer"]
    }
