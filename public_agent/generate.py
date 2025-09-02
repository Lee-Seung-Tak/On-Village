from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from public_agent.registry import tools
from public_agent.prompts import SYSTEM_PROMPT
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# === 1. LLM 정의 ===
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# === 2. 프롬프트 정의 (system + history + input + scratchpad) ===
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# === 3. Functions Agent 생성 ===
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# === 4. 세션별 히스토리 저장 ===
history_store = {}

def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

# === 5. 호출 함수 ===
def llm_generate(user_input: str, session_id: str = "default"):
    history = get_session_history(session_id)

    # 실행
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": history.messages,
    })

    # 결과
    output = response.get("output", "")

    # 히스토리에 추가
    history.add_message(HumanMessage(content=user_input))
    history.add_message(AIMessage(content=output))

    return output
