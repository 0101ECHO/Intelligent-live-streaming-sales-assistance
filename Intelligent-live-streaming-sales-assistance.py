"""
智能直播带货辅助系统 v2.0 — 抖音直播实时版
============================================
基于 LLM(DeepSeek-V4-Flash) + Agent + RAG + LangGraph + LangChain

功能：
  1. 实时接入抖音直播弹幕（浏览器注入脚本 + 弹窗桥接模式）
  2. 实时情感分析 + 意图识别（LLM）
  3. RAG 检索商品知识库 / 话术策略库
  4. Agent 智能决策最佳话术策略
  5. 实时生成推荐话术，主播即刻可用
  6. 数据看板：弹幕统计、情感分布、意图分析

运行模式：
  python Intelligent-live-streaming-sales-assistance.py              # 演示模式
  python Intelligent-live-streaming-sales-assistance.py --interactive  # 交互模式
  python Intelligent-live-streaming-sales-assistance.py --live         # 抖音实时接入模式（弹窗桥接）
"""

import os
import json
import re
import time
import asyncio
import threading
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ─── LangChain 核心 ───────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

# ─── DeepSeek LLM ─────────────────────────────────────────────────
from langchain_deepseek import ChatDeepSeek

# ─── LangGraph ────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END

# ─── LangChain Agent ──────────────────────────────────────────────
from langchain.agents import create_agent

# ─── WebSocket 服务 ──────────────────────────────────────────────
from websockets.asyncio.server import serve
import websockets.exceptions

# ─── HTTP 服务器（桥接服务） ────────────────────────────────────
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import urllib.parse

# ─── 加载环境变量 ─────────────────────────────────────────────────
load_dotenv()

# ===================================================================
# 第一部分：LLM 初始化
# ===================================================================

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.3,
    max_tokens=2048,
)

# ===================================================================
# 第二部分：RAG 知识库（商品知识 + 话术策略）
# ===================================================================

PRODUCT_KNOWLEDGE_BASE = [
    {
        "id": "prod_001", "name": "智能直播补光灯", "category": "直播设备",
        "price": 299,
        "features": ["360°环形灯珠", "三档色温调节", "手机支架一体式", "USB充电"],
        "target_audience": ["主播", "短视频创作者", "直播爱好者"],
        "selling_points": [
            "专业级补光效果，让皮肤显白显嫩",
            "一灯三用：补光、支架、充电宝",
            "性价比超高，同品质产品价格减半",
        ],
        "common_questions": {
            "亮度够不够": "60W高亮度，支持无级调光，从暗到亮随心调节",
            "续航多久": "内置5000mAh电池，满电可使用4-6小时",
            "支持什么手机": "6-10英寸手机通用，平板也能用",
        },
    },
    {
        "id": "prod_002", "name": "无线领夹麦克风", "category": "音频设备",
        "price": 199,
        "features": ["一键降噪", "10米远距离收音", "Type-C充电", "兼容多平台"],
        "target_audience": ["主播", "网课教师", "会议人员", "Vlog创作者"],
        "selling_points": [
            "AI智能降噪，嘈杂环境也能清晰收音",
            "即插即用，无需繁琐配对",
            "续航长达8小时，全天直播无忧",
        ],
        "common_questions": {
            "有杂音吗": "采用DSP智能降噪芯片，自动过滤环境噪音",
            "苹果手机能用吗": "支持iOS/Android/Windows全平台",
            "收音距离": "无障碍物情况下最远10米稳定收音",
        },
    },
    {
        "id": "prod_003", "name": "手机直播支架", "category": "直播设备",
        "price": 89,
        "features": ["伸缩高度1.7m", "蓝牙遥控", "360°旋转", "三脚架稳固"],
        "target_audience": ["主播", "直播带货", "视频拍摄"],
        "selling_points": [
            "加厚不锈钢管，承重更强不易倒",
            "蓝牙遥控拍照，远距离也能控制",
            "收纳仅40cm，出门携带超方便",
        ],
        "common_questions": {
            "稳不稳": "加宽三角支撑设计，承重可达3kg",
            "蓝牙怎么连接": "打开蓝牙遥控器，手机搜索配对即可",
            "高度多少": "伸缩范围40cm-170cm，坐着站着都能用",
        },
    },
]

STRATEGY_KNOWLEDGE_BASE = [
    {
        "id": "strategy_001", "name": "痛点共鸣法",
        "scenario": "用户对产品功能有疑虑",
        "description": "先认可用户的痛点，再提出解决方案",
        "template": "姐妹你说的太对了！我以前也遇到过这个问题...（共情）后来我发现...（解决方案）用了这个之后...（效果）",
        "effectiveness": "高",
    },
    {
        "id": "strategy_002", "name": "限时优惠法",
        "scenario": "用户犹豫价格",
        "description": "制造紧迫感，强调限时优惠",
        "template": "今天直播间专属福利价，只有最后XX件，抢完就恢复原价了！",
        "effectiveness": "极高",
    },
    {
        "id": "strategy_003", "name": "信任背书法",
        "scenario": "用户对品质有疑虑",
        "description": "引用权威数据、用户评价或销量数据",
        "template": "这个产品已经卖了XX万单，好评率XX%，你可以看看评论区...",
        "effectiveness": "高",
    },
    {
        "id": "strategy_004", "name": "场景代入法",
        "scenario": "产品功能展示",
        "description": "描述使用场景，让用户想象自己使用时的感受",
        "template": "想象一下，你拿到手之后...（场景描述）有了它，你再也不用...（痛点解决）",
        "effectiveness": "中",
    },
    {
        "id": "strategy_005", "name": "对比突出法",
        "scenario": "需要突出产品优势",
        "description": "与同类产品对比，突出本产品优势",
        "template": "市面上同价位的产品一般都是...（缺点）但是我们的...（优点）",
        "effectiveness": "中",
    },
    {
        "id": "strategy_006", "name": "销量冲刺法",
        "scenario": "直播即将结束或销量接近目标",
        "description": "用销量目标激励用户下单",
        "template": "今天我们的目标是XX单，现在已经XX单了，还差最后一波，大家一起冲！",
        "effectiveness": "极高",
    },
]


class SimpleRAG:
    """简易 RAG 检索系统（关键词匹配，无需外部向量数据库）"""

    def __init__(self):
        self.products = PRODUCT_KNOWLEDGE_BASE
        self.strategies = STRATEGY_KNOWLEDGE_BASE

    def search_products(self, query: str, top_k: int = 3) -> List[Dict]:
        query_lower = query.lower()
        scores = []
        for product in self.products:
            score = 0
            search_text = (
                f"{product['name']} {product['category']} "
                f"{' '.join(product['features'])} "
                f"{' '.join(product['selling_points'])}"
            ).lower()
            keywords = [kw for kw in re.findall(r'[\w]+', query_lower) if len(kw) >= 2]
            for kw in keywords:
                if kw in search_text:
                    score += 1
                for q, a in product['common_questions'].items():
                    if kw in q.lower() or kw in a.lower():
                        score += 0.5
            if score > 0:
                scores.append((score, product))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scores[:top_k]]

    def search_strategies(self, query: str, top_k: int = 2) -> List[Dict]:
        query_lower = query.lower()
        scores = []
        for strategy in self.strategies:
            score = 0
            search_text = (
                f"{strategy['name']} {strategy['scenario']} "
                f"{strategy['description']}"
            ).lower()
            keywords = [kw for kw in re.findall(r'[\w]+', query_lower) if len(kw) >= 2]
            for kw in keywords:
                if kw in search_text:
                    score += 1
            if score > 0:
                scores.append((score, strategy))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scores[:top_k]]


rag_system = SimpleRAG()

# ===================================================================
# 第三部分：Agent 工具定义
# ===================================================================

@tool
def query_product_info(query: str) -> str:
    """查询商品信息，包括价格、功能、卖点、常见问题等。"""
    products = rag_system.search_products(query)
    if not products:
        return "未找到相关商品信息。"
    results = []
    for p in products:
        info = (
            f"【{p['name']}】\n价格：¥{p['price']}\n类别：{p['category']}\n"
            f"功能特点：{'、'.join(p['features'])}\n"
            f"核心卖点：{'；'.join(p['selling_points'])}\n常见问题：\n"
        )
        for q, a in p['common_questions'].items():
            info += f"  Q: {q}\n  A: {a}\n"
        results.append(info)
    return "\n".join(results)


@tool
def analyze_danmaku_sentiment(danmaku_text: str) -> str:
    """分析弹幕的情感倾向和用户意图，返回JSON。"""
    prompt = f"""你是一个直播弹幕分析专家。请分析以下弹幕内容，返回JSON格式的分析结果。

弹幕内容：{danmaku_text}

请分析以下维度并返回JSON（不要包含其他内容）：
1. "情感倾向": "正面" / "负面" / "中性"
2. "用户意图": "询问价格" / "询问功能" / "抱怨" / "赞美" / "对比" / "犹豫" / "催促下单" / "其他"
3. "紧急程度": "高" / "中" / "低"
4. "购买意向": true / false
5. "简要分析": "一句话总结"
"""
    response = llm.invoke([
        SystemMessage(content="你是一个专业的直播弹幕分析助手，只返回JSON格式结果。"),
        HumanMessage(content=prompt),
    ])
    text = response.content
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            json.loads(json_match.group())
            return json_match.group()
        except json.JSONDecodeError:
            pass
    return json.dumps({
        "情感倾向": "中性", "用户意图": "其他",
        "紧急程度": "中", "购买意向": False, "简要分析": "分析失败，返回默认值"
    }, ensure_ascii=False)


@tool
def recommend_strategy(danmaku_analysis: str, product_info: str) -> str:
    """根据弹幕分析和商品信息，推荐最合适的话术策略。"""
    strategies = rag_system.search_strategies(danmaku_analysis)
    strategy_text = ""
    for s in strategies:
        strategy_text += f"策略：{s['name']}\n场景：{s['scenario']}\n描述：{s['description']}\n模板：{s['template']}\n\n"
    prompt = f"""你是一个直播带货话术策略专家。根据以下信息，推荐最合适的话术策略。

弹幕分析结果：{danmaku_analysis}
商品信息：{product_info}
可参考策略库：{strategy_text if strategy_text else "无匹配策略，请根据经验推荐"}

请返回JSON格式（不要包含其他内容）：
1. "推荐策略名称": ""
2. "推荐理由": ""
3. "具体话术": "可直接用于直播的完整话术"
4. "注意事项": ""
"""
    response = llm.invoke([
        SystemMessage(content="你是一个专业的直播带货话术策略顾问。"),
        HumanMessage(content=prompt),
    ])
    return response.content

# ===================================================================
# 第四部分：LangGraph 状态定义 & 节点
# ===================================================================

class LiveStreamState(TypedDict):
    danmaku_queue: List[Dict[str, Any]]
    current_danmaku: Optional[Dict[str, Any]]
    danmaku_history: List[Dict[str, Any]]
    danmaku_analysis: Optional[str]
    sentiment: Optional[str]
    user_intent: Optional[str]
    retrieved_products: List[Dict]
    retrieved_strategies: List[Dict]
    agent_decision: Optional[str]
    recommended_strategy: Optional[str]
    recommended_script: Optional[str]
    current_product: Optional[str]
    error: Optional[str]
    is_processing: bool


def node_receive_danmaku(state: LiveStreamState) -> LiveStreamState:
    """节点1：接收弹幕"""
    q = state.get("danmaku_queue", [])
    if not q:
        return {**state, "current_danmaku": None, "is_processing": False, "error": "没有待处理的弹幕"}
    current = q.pop(0)
    history = state.get("danmaku_history", [])
    history.append(current)
    return {**state, "danmaku_queue": q, "current_danmaku": current, "danmaku_history": history, "is_processing": True, "error": None}


def node_analyze_danmaku(state: LiveStreamState) -> LiveStreamState:
    """节点2：LLM 情感分析"""
    current = state.get("current_danmaku")
    if not current:
        return {**state, "error": "没有弹幕可分析"}
    text = current.get("content", "")
    if not text:
        return {**state, "error": "弹幕内容为空"}
    try:
        result = analyze_danmaku_sentiment.invoke({"danmaku_text": text})
        sentiment, intent = "中性", "其他"
        try:
            j = json.loads(result)
            sentiment = j.get("情感倾向", "中性")
            intent = j.get("用户意图", "其他")
        except Exception:
            pass
        return {**state, "danmaku_analysis": result, "sentiment": sentiment, "user_intent": intent, "error": None}
    except Exception as e:
        return {**state, "error": f"弹幕分析失败: {str(e)}"}


def node_rag_retrieve(state: LiveStreamState) -> LiveStreamState:
    """节点3：RAG 检索"""
    current = state.get("current_danmaku")
    analysis = state.get("danmaku_analysis", "")
    text = current.get("content", "") if current else ""
    query = text
    if analysis:
        try:
            intent = json.loads(analysis).get("用户意图", "")
            query = f"{text} {intent}"
        except Exception:
            pass
    products = rag_system.search_products(query)
    strategies = rag_system.search_strategies(query)
    return {**state, "retrieved_products": products, "retrieved_strategies": strategies, "error": None}


def node_agent_decision(state: LiveStreamState) -> LiveStreamState:
    """节点4：Agent 策略决策"""
    analysis = state.get("danmaku_analysis", "无分析结果")
    products = state.get("retrieved_products", [])
    strategies = state.get("retrieved_strategies", [])
    product_info = "暂无相关商品"
    if products:
        product_info = "\n".join([f"商品：{p['name']} | ¥{p['price']} | 卖点：{'；'.join(p['selling_points'])}" for p in products])
    strategy_info = "暂无匹配策略"
    if strategies:
        strategy_info = "\n".join([f"策略：{s['name']} | 场景：{s['scenario']} | 模板：{s['template']}" for s in strategies])
    try:
        agent = create_agent(
            model=llm, tools=[query_product_info, analyze_danmaku_sentiment],
            system_prompt="你是一个直播带货策略决策专家。根据弹幕分析和商品信息，选择最合适的话术策略。",
        )
        result = agent.invoke({"messages": [{"role": "user", "content": f"弹幕分析：{analysis}\n商品：{product_info}\n策略库：{strategy_info}\n请推荐最佳策略并给出话术建议。"}]})
        msgs = result.get("messages", [])
        decision = msgs[-1].content if msgs else "无决策结果"
        name = "综合策略推荐"
        for line in decision.split("\n"):
            if "策略" in line and "：" in line:
                name = line.split("：")[-1].strip()
                break
        return {**state, "agent_decision": decision, "recommended_strategy": name, "error": None}
    except Exception as e:
        return {**state, "error": f"Agent决策失败: {str(e)}"}


def node_generate_script(state: LiveStreamState) -> LiveStreamState:
    """节点5：生成话术"""
    strategy = state.get("agent_decision", "无策略")
    products = state.get("retrieved_products", [])
    text = state.get("current_danmaku", {}).get("content", "")
    sentiment = state.get("sentiment", "中性")
    intent = state.get("user_intent", "其他")
    product_info = "暂无相关商品"
    if products:
        product_info = "\n".join([f"商品：{p['name']}（¥{p['price']}）\n卖点：{'；'.join(p['selling_points'][:2])}" for p in products])
    ctx = f"弹幕：{text}\n情感：{sentiment}\n意图：{intent}"
    prompt = f"""你是一个顶级的直播带货主播。请根据以下信息，生成一段自然、有感染力、转化率高的直播话术。

策略参考：{strategy}
商品信息：{product_info}
当前弹幕上下文：{ctx}

要求：
1. 话术要自然口语化，像真人主播在说话
2. 要有互动感，直接回应弹幕中的问题或情绪
3. 要包含明确的行动号召（如"点击下方小黄车"）
4. 要突出产品的核心卖点
5. 时长控制在30-60秒
6. 使用亲切的称呼（如"宝宝们"、"家人们"）
"""
    try:
        response = llm.invoke([
            SystemMessage(content="你是一个经验丰富的直播带货主播，擅长转化话术。"),
            HumanMessage(content=prompt),
        ])
        cp = products[0]["name"] if products else state.get("current_product", "未知")
        return {**state, "recommended_script": response.content, "current_product": cp, "error": None}
    except Exception as e:
        return {**state, "error": f"话术生成失败: {str(e)}"}


# ===================================================================
# 第五部分：构建 LangGraph 工作流
# ===================================================================

def build_workflow():
    workflow = StateGraph(LiveStreamState)
    workflow.add_node("receive_danmaku", node_receive_danmaku)
    workflow.add_node("analyze_danmaku", node_analyze_danmaku)
    workflow.add_node("rag_retrieve", node_rag_retrieve)
    workflow.add_node("agent_decision", node_agent_decision)
    workflow.add_node("generate_script", node_generate_script)
    workflow.add_edge(START, "receive_danmaku")
    workflow.add_edge("receive_danmaku", "analyze_danmaku")
    workflow.add_edge("analyze_danmaku", "rag_retrieve")
    workflow.add_edge("rag_retrieve", "agent_decision")
    workflow.add_edge("agent_decision", "generate_script")
    workflow.add_edge("generate_script", END)
    return workflow.compile()

# ===================================================================
# 第六部分：核心引擎
# ===================================================================

class LiveStreamEngine:
    """直播辅助引擎 - 处理弹幕的完整工作流"""

    def __init__(self):
        self.workflow = build_workflow()
        self.state = self._initial_state()
        self.session_history = []

    def _initial_state(self) -> LiveStreamState:
        return {
            "danmaku_queue": [], "current_danmaku": None, "danmaku_history": [],
            "danmaku_analysis": None, "sentiment": None, "user_intent": None,
            "retrieved_products": [], "retrieved_strategies": [],
            "agent_decision": None, "recommended_strategy": None,
            "recommended_script": None, "current_product": None,
            "error": None, "is_processing": False,
        }

    def reset(self, product: str = None):
        self.state = self._initial_state()
        self.state["current_product"] = product
        self.session_history = []

    def process(self, danmaku_text: str, user: str = "观众") -> Dict[str, Any]:
        dm = {
            "id": f"dm_{int(time.time()*1000)}_{len(self.state['danmaku_history'])}",
            "content": danmaku_text, "user": user,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.state["danmaku_queue"].append(dm)
        try:
            self.state = self.workflow.invoke(self.state)
            self.session_history.append({
                "timestamp": datetime.now().isoformat(),
                "danmaku": danmaku_text,
                "analysis": self.state.get("danmaku_analysis"),
                "strategy": self.state.get("recommended_strategy"),
                "script": self.state.get("recommended_script"),
            })
            return {
                "success": True, "danmaku": danmaku_text,
                "analysis": self.state.get("danmaku_analysis"),
                "sentiment": self.state.get("sentiment"),
                "intent": self.state.get("user_intent"),
                "strategy": self.state.get("recommended_strategy"),
                "script": self.state.get("recommended_script"),
                "error": self.state.get("error"),
            }
        except Exception as e:
            return {"success": False, "danmaku": danmaku_text, "error": str(e)}

    def get_analytics(self) -> Dict:
        if not self.session_history:
            return {"message": "暂无数据"}
        sentiments, intents = [], []
        for r in self.session_history:
            if r.get("analysis"):
                try:
                    j = json.loads(r["analysis"])
                    sentiments.append(j.get("情感倾向", "未知"))
                    intents.append(j.get("用户意图", "未知"))
                except Exception:
                    pass
        return {
            "total": len(self.session_history),
            "sentiment": {"正面": sentiments.count("正面"), "负面": sentiments.count("负面"), "中性": sentiments.count("中性")},
            "intent": {i: intents.count(i) for i in set(intents)},
            "recent": self.session_history[-5:],
        }

# ===================================================================
# 第七部分：抖音实时接入 — 弹窗桥接模式
# ===================================================================

# ─── 抖音弹幕捕获脚本（浏览器控制台注入）v3.0 ─────────────────
# 使用弹窗桥接模式：window.open() + postMessage 跨域通信，
# 避免 HTTPS 页面 CSP 限制无法连接 ws:// 或 fetch http://localhost
DOUYIN_BROWSER_SCRIPT = r'''
// ============================================================
// 抖音直播弹幕捕获脚本 v3.0 (弹窗桥接模式)
// 使用方法：在抖音直播间网页 (https://live.douyin.com/xxx) 按 F12
// 打开控制台(Console)，粘贴以下代码后回车
// ============================================================
// 原理：通过 window.open 打开桥接页面（HTTP 页面不受 CSP 限制），
// 桥接页面连接 WebSocket，主页面和桥接页面通过 postMessage 通信
// ============================================================
(function() {
    const BRIDGE_URL = "http://localhost:8080/bridge.html";
    let bridgeWindow = null;
    let sentTexts = new Set();
    let sendCount = 0;
    let bridgeCheckTimer = null;

    // 打开桥接窗口
    function openBridge() {
        if (bridgeWindow && !bridgeWindow.closed) {
            return true;
        }
        try {
            bridgeWindow = window.open(
                BRIDGE_URL,
                "LiveStreamBridge",
                "width=450,height=650,menubar=no,toolbar=no,location=no,status=no,scrollbars=yes"
            );
            if (!bridgeWindow || bridgeWindow.closed) {
                console.warn("%c[⚠️ 弹幕捕获] 桥接窗口被阻止！请允许弹出窗口后重新运行脚本", "color:red;font-size:14px;font-weight:bold");
                return false;
            }
            return true;
        } catch(e) {
            console.error("[弹幕捕获] 打开桥接窗口失败:", e);
            return false;
        }
    }

    // 发送弹幕到桥接窗口
    function sendToBridge(text) {
        if (!text || sentTexts.has(text)) return;
        sentTexts.add(text);
        sendCount++;

        if (bridgeWindow && !bridgeWindow.closed) {
            bridgeWindow.postMessage({
                type: 'danmaku',
                content: text,
                timestamp: new Date().toISOString()
            }, '*');
        }
    }

    // 监听桥接页面返回的分析结果
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'assistant_result') {
            var r = event.data.data;
            console.log(
                "%c[🎯 弹幕分析] %c" + (r.sentiment||'?') + "%c | 意图: " + (r.intent||'?') + " | 策略: " + (r.strategy||'?'),
                "color:green;font-weight:bold",
                "color:blue",
                "color:#333"
            );
            if (r.script) {
                console.log("%c💬 推荐话术: %c" + r.script.substring(0, 100), "color:#c00;font-weight:bold", "color:#333");
            }
        }
    });

    // 定期检查桥接窗口状态
    function checkBridge() {
        if (!bridgeWindow || bridgeWindow.closed) {
            console.log("%c[🔄 弹幕捕获] 桥接窗口已关闭，尝试重新打开...", "color:orange");
            openBridge();
        }
    }

    // 开始捕获弹幕
    function startCapture() {
        console.log("%c[🎯 弹幕捕获] 开始监听弹幕 (桥接模式)...", "color:blue;font-size:14px");

        // 监听 DOM 变化（捕获新增的弹幕元素）
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1) {
                            try {
                                // 抖音弹幕常见选择器
                                var selectors = [
                                    '.webcast-chatroom__item__content',
                                    '.chatroom-content',
                                    '[class*="chatroom"] [class*="content"]',
                                    '.danmaku-item .text',
                                    '.chat-message-content',
                                    'span[class*="content"]',
                                    '[class*="message"] [class*="text"]',
                                    '[class*="chat"] [class*="msg"]',
                                ];
                                var danmakuEl = null;
                                for (var i = 0; i < selectors.length; i++) {
                                    danmakuEl = node.querySelector(selectors[i]);
                                    if (danmakuEl) break;
                                }
                                if (!danmakuEl) {
                                    for (var i = 0; i < selectors.length; i++) {
                                        try { if (node.matches(selectors[i])) { danmakuEl = node; break; } } catch(e) {}
                                    }
                                }
                                if (danmakuEl && danmakuEl.textContent.trim()) {
                                    sendToBridge(danmakuEl.textContent.trim());
                                }
                            } catch(e) {}
                        }
                    });
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
            characterData: true,
        });

        // 定时轮询（兜底方案）
        setInterval(function() {
            try {
                var selectors = [
                    '.webcast-chatroom__item__content',
                    '.chatroom-content',
                    '[class*="chatroom"] [class*="content"]',
                    '[class*="message"] [class*="text"]',
                ];
                for (var s = 0; s < selectors.length; s++) {
                    var els = document.querySelectorAll(selectors[s]);
                    for (var e = 0; e < els.length; e++) {
                        var text = els[e].textContent.trim();
                        if (text) sendToBridge(text);
                    }
                }
            } catch(e) {}
        }, 3000);

        // 每5秒检查桥接窗口状态
        bridgeCheckTimer = setInterval(checkBridge, 5000);
    }

    // 启动
    if (openBridge()) {
        setTimeout(startCapture, 2000);
        console.log("%c[🎯 抖音弹幕捕获器 v3.0 已加载]", "color:purple;font-size:16px;font-weight:bold");
        console.log("  🌉 桥接页面: " + BRIDGE_URL);
        console.log("  ✅ 弹幕将通过 postMessage → 桥接页面 → WebSocket 传输");
        console.log("  ⚠️ 如果弹出窗口被拦截，请点击'允许弹出窗口'后刷新页面重试");
        console.log("  🛑 刷新页面即可停止");
    } else {
        console.error("%c[❌ 弹幕捕获] 无法打开桥接窗口，请检查浏览器是否阻止了弹出窗口", "color:red;font-size:14px");
    }
})();
'''

# ─── 桥接 HTML 页面（弹窗加载，连接 WebSocket，跨域通信） ────────
BRIDGE_HTML = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>直播辅助桥接页面</title>
<style>
  body { font-family: sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }
  h2 { color: #e94560; }
  .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
  .connected { background: #1b5e20; }
  .disconnected { background: #b71c1c; }
  #log { max-height: 400px; overflow-y: auto; font-size: 13px; }
  .log-entry { padding: 4px 0; border-bottom: 1px solid #333; }
</style>
</head>
<body>
<h2>🎯 直播辅助桥接页面</h2>
<div id="status" class="status disconnected">⏳ 正在连接 WebSocket...</div>
<div id="stats">已发送: <span id="sentCount">0</span> | 已接收: <span id="recvCount">0</span></div>
<h3>📋 实时日志</h3>
<div id="log"><div class="log-entry">🔄 页面已加载，正在连接...</div></div>
<script>
(function() {
    const WS_URL = "ws://localhost:8765";
    let ws = null;
    let reconnectTimer = null;
    let sentCount = 0;
    let recvCount = 0;

    function addLog(msg) {
        var log = document.getElementById('log');
        var entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
    }

    function updateStatus(connected, msg) {
        var el = document.getElementById('status');
        el.className = 'status ' + (connected ? 'connected' : 'disconnected');
        el.textContent = msg;
    }

    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) return;
        try {
            ws = new WebSocket(WS_URL);
            ws.onopen = function() {
                updateStatus(true, '✅ WebSocket 已连接');
                addLog('✅ WebSocket 连接成功');
                ws.send(JSON.stringify({type: "system", message: "桥接页面已连接"}));
            };
            ws.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    if (data.type === 'result') {
                        recvCount++;
                        document.getElementById('recvCount').textContent = recvCount;
                        var r = data.data;
                        addLog('📨 结果: 情感=' + (r.sentiment||'?') + ' 意图=' + (r.intent||'?') + ' 策略=' + (r.strategy||'?'));
                        // 转发给父页面（抖音直播间页面）
                        if (window.opener && !window.closed) {
                            window.opener.postMessage({type: 'assistant_result', data: r}, '*');
                        }
                    }
                } catch(e) {}
            };
            ws.onclose = function() {
                updateStatus(false, '⚠️ 连接断开，5秒后重连...');
                addLog('⚠️ 连接断开，5秒后重连');
                reconnectTimer = setTimeout(connect, 5000);
            };
            ws.onerror = function() {
                updateStatus(false, '❌ 连接错误，5秒后重试...');
                addLog('❌ 连接错误，5秒后重试');
                if (reconnectTimer) clearTimeout(reconnectTimer);
                reconnectTimer = setTimeout(connect, 5000);
            };
        } catch(e) {
            addLog('❌ 创建连接失败: ' + e.message);
            reconnectTimer = setTimeout(connect, 5000);
        }
    }

    // 监听来自父页面（抖音直播间）的弹幕消息
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'danmaku') {
            sentCount++;
            document.getElementById('sentCount').textContent = sentCount;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'danmaku',
                    content: event.data.content,
                    timestamp: new Date().toISOString()
                }));
                addLog('📤 转发弹幕: ' + event.data.content.substring(0, 30));
            }
        }
    });

    connect();
    addLog('🔄 桥接页面就绪，等待 WebSocket 连接...');
})();
</script>
</body>
</html>'''

# ─── HTTP 桥接服务器（提供桥接页面） ────────────────────────────

class BridgeHTTPHandler(BaseHTTPRequestHandler):
    """HTTP 桥接请求处理器 - 提供桥接页面"""

    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/bridge', '/bridge.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(BRIDGE_HTML.encode('utf-8'))
        elif parsed.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            status = {
                "status": "running",
                "total_danmaku": live_analytics.get("total", 0),
                "buffer_size": len(live_danmaku_buffer),
                "engine_ready": live_engine is not None,
            }
            self.wfile.write(json.dumps(status, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(b'LiveStream Assistant Bridge Server v2.0\n')
            self.wfile.write(f'Use http://localhost:{self.server.server_port}/bridge.html for bridge page\n'.encode())

    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        """自定义日志，格式更简洁"""
        if len(args) >= 3:
            print(f"  [HTTP] {args[0]} {args[1]} {args[2]}")
        else:
            print(f"  [HTTP] {' '.join(str(a) for a in args)}")


def run_http_server(host: str = 'localhost', port: int = 8080):
    """启动 HTTP 桥接服务器（在独立线程中运行）"""
    server = HTTPServer((host, port), BridgeHTTPHandler)
    print(f"  🌐 HTTP 桥接服务启动: http://{host}:{port}")
    print(f"     📋 GET  /bridge.html - 桥接页面（弹窗方式，主推方案）")
    print(f"     📊 GET  /status      - 服务状态")
    server.serve_forever()


# ─── 全局实时数据缓冲区 ────────────────────────────────────────
live_danmaku_buffer: deque = deque(maxlen=200)
live_processed_results: deque = deque(maxlen=100)
live_analytics: Dict = {"total": 0, "sentiment": {}, "intent": {}}
live_engine: LiveStreamEngine = None


async def ws_handler(websocket):
    """WebSocket 连接处理器 - 接收来自桥接页面的弹幕"""
    global live_engine
    print(f"  🔗 桥接页面已连接: {websocket.remote_address}")

    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "danmaku":
                content = data.get("content", "").strip()
                if not content:
                    continue

                # 加入缓冲区
                live_danmaku_buffer.append({
                    "content": content,
                    "time": datetime.now().strftime("%H:%M:%S"),
                })

                # 实时处理
                if live_engine:
                    result = live_engine.process(content, user="抖音观众")
                    live_processed_results.append(result)

                    # 更新看板数据
                    if result.get("success"):
                        s = result.get("sentiment", "中性")
                        i = result.get("intent", "其他")
                        live_analytics["total"] = live_analytics.get("total", 0) + 1
                        live_analytics["sentiment"][s] = live_analytics["sentiment"].get(s, 0) + 1
                        live_analytics["intent"][i] = live_analytics["intent"].get(i, 0) + 1

                    # 打印实时结果到控制台
                    print_live_result(result)

                    # 将结果回传给桥接页面
                    await websocket.send(json.dumps({
                        "type": "result",
                        "data": {
                            "sentiment": result.get("sentiment"),
                            "intent": result.get("user_intent"),
                            "strategy": result.get("recommended_strategy"),
                            "script": result.get("recommended_script"),
                        }
                    }, ensure_ascii=False))

            elif msg_type == "system":
                print(f"  📡 系统消息: {data.get('message', '')}")

        except json.JSONDecodeError:
            continue
        except websockets.exceptions.ConnectionClosed:
            break


async def start_ws_server(host: str = "localhost", port: int = 8765):
    """启动 WebSocket 服务器（桥接页面使用）"""
    print(f"\n  🌐 WebSocket 服务已启动: ws://{host}:{port}（供桥接页面使用）")
    print(f"  {'=' * 60}")
    print(f"  📋 使用说明（弹窗桥接模式）:")
    print(f"  {'=' * 60}")
    print(f"  ")
    print(f"  ✅ 步骤一：打开抖音直播间网页")
    print(f"  ✅ 步骤二：按 F12 打开开发者工具 → Console")
    print(f"  ✅ 步骤三：复制下方脚本粘贴到 Console 后回车")
    print(f"  ✅ 步骤四：系统会自动弹出桥接窗口，弹幕开始实时传输！")
    print(f"  ")
    print(f"  💡 提示：")
    print(f"     - 如果浏览器拦截了弹出窗口，请点击'允许弹出窗口'")
    print(f"     - 然后刷新抖音页面，重新粘贴脚本")
    print(f"     - 桥接窗口可最小化，不要关闭")
    print(f"     - 如需手动查看桥接页面: http://localhost:8080/bridge.html")
    print(f"  ")
    print(f"  {'=' * 60}")
    print(f"  📜 浏览器注入脚本（复制下方全部内容，粘贴到 Console 后回车）:")
    print(f"  {'=' * 60}")
    print(DOUYIN_BROWSER_SCRIPT)
    print(f"  {'=' * 60}\n")

    async with serve(ws_handler, host, port):
        await asyncio.Future()  # 永久运行


def print_live_result(result: Dict[str, Any]):
    """实时打印处理结果到控制台（带颜色标记）"""
    if result["success"]:
        script = result.get("script", "")
        # 只显示前 100 个字
        preview = script[:120] + "..." if len(script) > 120 else script
        print(f"\n  ╔══ {'=' * 40}")
        print(f"  ║  📨 [弹幕] {result['danmaku']}")
        print(f"  ║  📊 [情感] {result.get('sentiment', '?')}  🎯 [意图] {result.get('intent', '?')}")
        print(f"  ║  💡 [策略] {result.get('strategy', '?')}")
        print(f"  ║  🗣️  [话术] {preview}")
        print(f"  ╚══ {'=' * 40}")
    else:
        print(f"  ╔══ ❌ 处理失败: {result.get('error', '?')}")


def live_dashboard():
    """实时看板（在另一个线程中定期刷新）"""
    while True:
        time.sleep(10)
        if live_analytics["total"] > 0:
            print(f"\n  📊 [实时看板] 总弹幕: {live_analytics['total']} | "
                  f"情感: {live_analytics['sentiment']} | "
                  f"意图: {live_analytics['intent']}")


def run_live_mode():
    """运行抖音实时直播辅助模式"""
    global live_engine

    print("=" * 60)
    print("  🎯 智能直播带货辅助系统 — 抖音实时模式 v2.0")
    print("  LLM: DeepSeek-V4-Flash | Agent | RAG | LangGraph | LangChain")
    print("=" * 60)

    # 初始化引擎
    print("\n🔄 初始化引擎...")
    live_engine = LiveStreamEngine()
    live_engine.reset("智能直播补光灯")
    print("✅ 引擎就绪！\n")

    # 启动 HTTP 桥接服务器（独立线程）
    print("🔄 启动 HTTP 桥接服务...")
    http_thread = threading.Thread(
        target=run_http_server,
        args=('localhost', 8080),
        daemon=True,
        name="http-bridge"
    )
    http_thread.start()
    print("✅ HTTP 桥接服务已启动\n")

    # 启动看板线程
    dashboard_thread = threading.Thread(target=live_dashboard, daemon=True)
    dashboard_thread.start()

    # 启动 WebSocket 服务（阻塞，核心服务）
    asyncio.run(start_ws_server())


# ===================================================================
# 第八部分：演示 / 交互模式
# ===================================================================

def print_header():
    print("=" * 60)
    print("  🎯 智能直播带货辅助系统 v2.0")
    print("  LLM: DeepSeek-V4-Flash | Agent | RAG | LangGraph | LangChain")
    print("=" * 60)
    print()


def print_result(result: Dict[str, Any]):
    if result["success"]:
        print(f"\n📝 [弹幕] {result['danmaku']}")
        print(f"📊 [情感] {result.get('sentiment', '未知')}  |  🎯 [意图] {result.get('intent', '未知')}")
        print(f"💡 [策略] {result.get('strategy', '无')}")
        if result.get("error"):
            print(f"⚠️  [提示] {result['error']}")
        print(f"\n🗣️  [推荐话术]")
        print("─" * 40)
        print(result.get("script", "生成失败"))
        print("─" * 40)
    else:
        print(f"❌ 处理失败: {result.get('error', '未知错误')}")


def demo_scenario():
    """演示模式"""
    print_header()
    engine = LiveStreamEngine()
    engine.reset("智能直播补光灯")
    test_danmaku = [
        "这个灯亮度够不够啊？",
        "价格有点贵了，能不能便宜点",
        "和其他牌子比有什么优势？",
        "质量怎么样，耐不耐用？",
    ]
    print("🎭 开始模拟直播弹幕处理...\n")
    for i, dm in enumerate(test_danmaku, 1):
        print(f"{'─' * 60}")
        print(f"  📨 [弹幕 #{i}] {dm}")
        print(f"{'─' * 60}")
        result = engine.process(dm)
        print_result(result)
    print(f"\n{'=' * 60}")
    print("  📊 会话分析报告")
    print(f"{'=' * 60}")
    analytics = engine.get_analytics()
    print(f"  总处理弹幕数: {analytics.get('total', 0)}")
    print(f"  情感分布: {analytics.get('sentiment', {})}")
    print(f"  意图分布: {analytics.get('intent', {})}")
    print(f"{'=' * 60}")
    print("\n✅ 演示完成！")


def interactive_mode():
    """交互模式"""
    print_header()
    engine = LiveStreamEngine()
    engine.reset()
    print("💡 输入弹幕内容开始分析，输入 'exit' 退出，输入 'report' 查看分析报告\n")
    while True:
        try:
            dm = input("🎤 弹幕 > ").strip()
            if dm.lower() == "exit":
                print("👋 感谢使用！")
                break
            elif dm.lower() == "report":
                a = engine.get_analytics()
                print(f"\n📊 分析报告: 总处理: {a.get('total', 0)} | 情感: {a.get('sentiment', {})} | 意图: {a.get('intent', {})}")
                continue
            elif not dm:
                continue
            result = engine.process(dm)
            print_result(result)
            print()
        except KeyboardInterrupt:
            print("\n👋 感谢使用！")
            break
        except Exception as e:
            print(f"❌ 错误: {str(e)}")


# ===================================================================
# 第九部分：程序入口
# ===================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--live":
            run_live_mode()
        elif mode == "--interactive":
            interactive_mode()
        else:
            demo_scenario()
    else:
        demo_scenario()