"""简化的主程序模块

该模块负责：
- FastAPI应用初始化
- 路由定义
- 请求处理逻辑
- 模块间协调
"""

import os
import json
import time
import hashlib
import asyncio
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import numpy as np
from openai import AsyncOpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import urllib.parse
import glob
from datetime import datetime

from config import config
from document_manager import document_processor, DocumentLoader
from retrieval_methods import RetrievalManager
from database import db
from mcp_api import router as mcp_router
from L0_agent.L0_agent import LangGraphAgent
# 添加 GraphRAG 相关导入
from L1_agent_rag.config import GraphRAGConfig
from L1_agent_rag.L1_agent_rag import L1AgentRAG

# 全局变量
client: Optional[AsyncOpenAI] = None
retrieval_manager: Optional[RetrievalManager] = None
document_loader = DocumentLoader()

# 创建全局agent实例
agent = LangGraphAgent()

# 模板和静态文件
templates = Jinja2Templates(directory="templates")

def create_optimized_client(model_name: str) -> AsyncOpenAI:
    """创建优化的OpenAI客户端
    
    参数:
        model_name: 模型名称
        
    返回:
        OpenAI客户端实例
    """
    model_config = config.get_model_config(model_name)
    if not model_config:
        raise ValueError(f"未找到模型配置: {model_name}")
    
    return AsyncOpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["api_base"],
        timeout=30.0,
        max_retries=3
    )

async def generate_answer(query: str, related_info: str = "", 
                         web_search_results: str = "", 
                         conversation_history: List[Dict] = None,
                         model_name: str = None) -> str:
    """生成答案
    
    参数:
        query: 用户查询
        related_info: 相关信息
        web_search_results: 网络搜索结果
        conversation_history: 对话历史
        model_name: 模型名称
        
    返回:
        生成的答案
    """
    global client
    
    if model_name is None:
        model_name = config.system_config["default_model"]
    
    # 动态创建客户端（每次都重新创建以支持模型切换）
    client = create_optimized_client(model_name)
    
    model_config = config.get_model_config(model_name)
    if not model_config:
        return "模型配置错误"
    
    # 构建上下文
    context_parts = []
    
    if conversation_history:
        context_parts.append("对话历史：")
        for msg in conversation_history[-3:]:  # 只取最近3轮对话
            role = "用户" if msg['role'] == 'user' else "助手"
            context_parts.append(f"{role}: {msg['content']}")
        context_parts.append("")
    
    if web_search_results:
        context_parts.append("网络搜索结果：")
        context_parts.append(web_search_results)
        context_parts.append("")
    
    if related_info:
        context_parts.append("相关文档信息：")
        context_parts.append(related_info)
        context_parts.append("")
    
    context = "\n".join(context_parts)
    
    # 构建提示词
    if web_search_results and not related_info:
        prompt = f"""{context}用户问题: {query}

请基于上述网络搜索结果回答用户的问题。如果搜索结果不足以回答问题，请说明需要更多信息。"""
    elif related_info and not web_search_results:
        prompt = f"""{context}用户问题: {query}

请基于上述文档信息回答用户的问题。如果文档信息不足以回答问题，请说明。"""
    elif web_search_results and related_info:
        prompt = f"""{context}用户问题: {query}

请综合上述网络搜索结果和文档信息回答用户的问题。"""
    else:
        prompt = f"""{context}用户问题: {query}

请回答用户的问题。"""
    
    try:
        response = await client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": "你是一个智能助手，能够基于提供的信息准确回答用户问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            stream=False
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"生成答案时出错: {str(e)}"

async def generate_streaming_answer(query: str, related_info: str = "", 
                                  web_search_results: str = "", 
                                  conversation_history: List[Dict] = None,
                                  model_name: str = None):
    """生成流式答案
    
    参数:
        query: 用户查询
        related_info: 相关信息
        web_search_results: 网络搜索结果
        conversation_history: 对话历史
        model_name: 模型名称
        
    返回:
        流式响应生成器
    """
    global client
    
    if model_name is None:
        model_name = config.system_config["default_model"]
    
    # 动态创建客户端（每次都重新创建以支持模型切换）
    client = create_optimized_client(model_name)
    
    model_config = config.get_model_config(model_name)
    if not model_config:
        yield f"data: {json.dumps({'error': '模型配置错误'}, ensure_ascii=False)}\n\n"
        return
    
    # 构建上下文（与generate_answer相同的逻辑）
    context_parts = []
    
    if conversation_history:
        context_parts.append("对话历史：")
        for msg in conversation_history[-3:]:
            role = "用户" if msg['role'] == 'user' else "助手"
            context_parts.append(f"{role}: {msg['content']}")
        context_parts.append("")
    
    if web_search_results:
        context_parts.append("网络搜索结果：")
        context_parts.append(web_search_results)
        context_parts.append("")
    
    if related_info:
        context_parts.append("相关文档信息：")
        context_parts.append(related_info)
        context_parts.append("")
    
    context = "\n".join(context_parts)
    
    # 构建提示词
    if web_search_results and not related_info:
        prompt = f"""{context}用户问题: {query}

请基于上述网络搜索结果回答用户的问题。如果搜索结果不足以回答问题，请说明需要更多信息。"""
    elif related_info and not web_search_results:
        prompt = f"""{context}用户问题: {query}

请基于上述文档信息回答用户的问题。如果文档信息不足以回答问题，请说明。"""
    elif web_search_results and related_info:
        prompt = f"""{context}用户问题: {query}

请综合上述网络搜索结果和文档信息回答用户的问题。"""
    else:
        prompt = f"""{context}用户问题: {query}

请回答用户的问题。"""
    
    try:
        response = await client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": "你是一个智能助手，能够基于提供的信息准确回答用户问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
        
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': f'生成答案时出错: {str(e)}'}, ensure_ascii=False)}\n\n"

async def quick_system_check() -> bool:
    """快速检查系统状态
    
    返回:
        系统是否正常
    """
    return document_processor.initialized

async def initialize_system():
    """初始化系统"""
    global retrieval_manager
    
    print("正在初始化系统...")
    
    # 检查Faiss索引文件是否存在
    faiss_dir = Path("storage/Faiss")
    faiss_index_path = faiss_dir / "faiss_index.faiss"
    chunks_mapping_path = faiss_dir / "chunks_mapping.npy"
    
    if not faiss_index_path.exists() or not chunks_mapping_path.exists():
        print("检测到Faiss索引文件缺失，开始重新创建索引...")
        print(f"faiss_index.faiss 存在: {faiss_index_path.exists()}")
        print(f"chunks_mapping.npy 存在: {chunks_mapping_path.exists()}")
        
        # 初始化文档处理器
        if not document_processor.initialize():
            print("文档处理器初始化失败")
            return False
        
        # 重新处理所有文档以创建索引
        print("开始处理文档并创建索引...")
        if not document_processor.process_documents():
            print("文档处理和索引创建失败")
            return False
        
        print("Faiss索引创建完成")
    else:
        print("检测到Faiss索引文件已存在，直接加载...")
        # 初始化文档处理器
        if not document_processor.initialize():
            print("文档处理器初始化失败")
            return False
    
    # 检查GraphRAG多图谱是否存在
    graph_config = GraphRAGConfig()
    
    if not graph_config.has_existing_graphs():
        print("检测到知识图谱文件缺失，开始创建知识图谱...")
        print(f"多图谱目录: {graph_config.get_multi_graphs_dir()}")
        
        l1_agent = None
        try:
            # 初始化L1AgentRAG
            l1_agent = L1AgentRAG()
            
            # 构建知识图谱
            print("开始构建知识图谱...")
            success = await l1_agent.build_knowledge_graph(force_rebuild=True)
            
            if success:
                print("知识图谱创建完成")
            else:
                print("知识图谱创建失败")
                # 不返回False，允许系统继续运行
        except Exception as e:
            print(f"知识图谱创建过程中出现异常: {str(e)}")
            # 不返回False，允许系统继续运行
        finally:
            # 确保清理资源
            if l1_agent is not None:
                try:
                    # 如果L1AgentRAG有清理方法，在这里调用
                    if hasattr(l1_agent, 'cleanup'):
                        await l1_agent.cleanup()
                except Exception as cleanup_error:
                    print(f"清理L1AgentRAG资源时出现异常: {str(cleanup_error)}")
                finally:
                    del l1_agent
    else:
        print("检测到知识图谱文件已存在，跳过创建...")
        # 显示已有图谱信息
        multi_graphs_dir = Path(graph_config.get_multi_graphs_dir())
        if multi_graphs_dir.exists():
            graph_files = list(multi_graphs_dir.glob("*.json"))
            print(f"已有图谱文件数量: {len(graph_files)}")
    
    # 初始化检索管理器
    retrieval_manager = RetrievalManager(document_processor)
    
    print("系统初始化完成")
    return True

async def reprocess_documents():
    """重新处理所有文档"""
    print("开始重新处理文档...")
    return document_processor.process_documents()

async def add_document_to_index(doc_path: str) -> bool:
    """异步增量添加文档到索引
    
    参数:
        doc_path: 文档路径
        
    返回:
        是否成功添加
    """
    return await document_processor.add_document_async(doc_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    await initialize_system()
    yield
    # 关闭时清理资源
    print("正在关闭应用，清理资源...")
    try:
        # 清理全局L1AgentRAGTool实例
        from L1_agent_rag.L1_agent_rag_tool import get_l1_agent_rag_tool
        l1_tool = get_l1_agent_rag_tool()
        if hasattr(l1_tool, 'cleanup'):
            await l1_tool.cleanup()
    except Exception as e:
        print(f"清理L1AgentRAGTool时出现异常: {str(e)}")
    
    try:
        # 清理数据库连接
        await db.close()
    except Exception as e:
        print(f"关闭数据库连接时出现异常: {str(e)}")
    
    print("应用资源清理完成")

# 创建FastAPI应用
app = FastAPI(
    title="企业知识库助手",
    description="基于RAG的智能问答系统",
    version="2.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 包含MCP路由
app.include_router(mcp_router)

# 设置上传目录
UPLOADS_DIR = Path(config.system_config["upload_dir"])
UPLOADS_DIR.mkdir(exist_ok=True)

# Pydantic模型
class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    model_name: Optional[str] = None
    specific_file: Optional[str] = None

class ConversationCreate(BaseModel):
    title: str

class ConversationUpdate(BaseModel):
    title: str

class ConversationStar(BaseModel):
    starred: bool

# 路由定义
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页"""
    conversations = await db.get_conversations(limit=20)
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "conversations": conversations}
    )

@app.get("/files")
async def get_files():
    """获取文件列表"""
    supported_extensions = ('.txt', '.pdf', '.docx', '.md', '.xlsx', '.xls')
    files = []
    
    for ext in supported_extensions:
        pattern = str(UPLOADS_DIR / f"*{ext}")
        files.extend([os.path.basename(f) for f in glob.glob(pattern)])
    
    return {"files": sorted(files)}

@app.get("/query/{question}")
async def query_endpoint(question: str, 
                        conversation_id: str = Query(None),
                        model_name: str = Query(None),
                        specific_file: str = Query(None)):
    """查询接口（使用LangGraph Agent）"""
    start_time = time.time()
    
    try:
        # 快速系统检查
        if not await quick_system_check():
            raise HTTPException(status_code=500, detail="系统未正确初始化")
        
        # URL解码
        question = urllib.parse.unquote(question)
        if specific_file:
            specific_file = urllib.parse.unquote(specific_file)
        
        print(f"\n=== LangGraph Agent 查询开始 ===")
        print(f"问题: {question}")
        print(f"对话ID: {conversation_id}")
        print(f"模型: {model_name or config.system_config['default_model']}")
        print(f"指定文件: {specific_file}")
        
        # 使用LangGraph Agent处理查询
        async def generate():
            async for chunk in agent.query(
                query=question,
                conversation_id=conversation_id,
                model_name=model_name,
                specific_file=specific_file
            ):
                yield chunk
        
        print(f"Agent查询设置耗时: {time.time() - start_time:.2f}秒")
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print(f"Agent查询处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/query/{question}")
async def agent_query_endpoint(question: str, 
                              conversation_id: str = Query(None),
                              model_name: str = Query(None),
                              specific_file: str = Query(None)):
    """LangGraph Agent查询接口"""
    start_time = time.time()
    
    try:
        # 快速系统检查
        if not await quick_system_check():
            raise HTTPException(status_code=500, detail="系统未正确初始化")
        
        # URL解码
        question = urllib.parse.unquote(question)
        if specific_file:
            specific_file = urllib.parse.unquote(specific_file)
        
        print(f"\n=== LangGraph Agent 查询开始 ===")
        print(f"问题: {question}")
        print(f"对话ID: {conversation_id}")
        print(f"模型: {model_name or config.system_config['default_model']}")
        print(f"指定文件: {specific_file}")
        
        # 使用LangGraph Agent处理查询
        async def generate():
            async for chunk in agent.query(
                query=question,
                conversation_id=conversation_id,
                model_name=model_name,
                specific_file=specific_file
            ):
                yield chunk
        
        print(f"Agent查询设置耗时: {time.time() - start_time:.2f}秒")
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print(f"Agent查询处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件"""
    try:
        # 检查文件类型
        allowed_extensions = {'.txt', '.pdf', '.docx', '.md', '.xlsx', '.xls'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_ext}"
            )
        
        # 保存文件
        file_path = UPLOADS_DIR / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 增量添加到索引
        success = await add_document_to_index(str(file_path))
        
        if success:
            return {"message": f"文件 {file.filename} 上传并索引成功"}
        else:
            return {"message": f"文件 {file.filename} 上传成功，但索引失败"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.post("/api/documents/rebuild")
async def rebuild_documents():
    """手动重建文档索引"""
    try:
        print("收到手动重建文档索引请求")
        success = await reprocess_documents()
        
        if success:
            return {"message": "文档索引重建成功"}
        else:
            return {"message": "文档索引重建失败", "error": True}
            
    except Exception as e:
        print(f"重建文档索引出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重建失败: {str(e)}")

@app.get("/api/documents/status")
async def get_document_status():
    """获取文档处理状态"""
    try:
        return {
            "initialized": document_processor.initialized,
            "rebuilding": getattr(document_processor, '_rebuilding', False),
            "total_chunks": len(document_processor.all_chunks) if document_processor.all_chunks else 0,
            "total_documents": len(document_processor.doc_sources) if document_processor.doc_sources else 0,
            "last_check": getattr(document_processor, '_last_document_check', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@app.get("/documents")
async def get_documents():
    """获取文档列表"""
    try:
        files = []
        for file_path in UPLOADS_DIR.glob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return {"documents": sorted(files, key=lambda x: x["modified"], reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

# 对话管理路由
@app.get("/api/conversations")
async def get_conversations():
    """获取对话列表"""
    try:
        conversations = await db.get_conversations(limit=50)
        result = []
        for conv in conversations:
            result.append({
                "conversation_id": conv[0],  # 修复字段名匹配问题
                "title": conv[1],
                "starred": conv[2],  # 保持原始数值，前端会处理转换
                "created_at": conv[3],
                "updated_at": conv[4]
            })
        return {"conversations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {str(e)}")

@app.get("/api/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    """获取对话详情"""
    try:
        messages = await db.get_messages(conversation_id)
        result = []
        for msg in messages:
            result.append({
                "id": msg[0],
                "content": msg[1],
                "type": msg[2],
                "created_at": msg[3]
            })
        return {"messages": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话详情失败: {str(e)}")

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """删除对话"""
    try:
        success = await db.delete_conversation(conversation_id)
        if success:
            return {"message": "对话删除成功"}
        else:
            raise HTTPException(status_code=404, detail="对话不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除对话失败: {str(e)}")

@app.post("/api/conversations")
async def create_conversation_endpoint(request: ConversationCreate):
    """创建新对话"""
    try:
        conversation_id = await db.create_conversation(request.title)
        return {"conversation_id": conversation_id, "message": "对话创建成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建对话失败: {str(e)}")

@app.put("/api/conversations/{conversation_id}")
async def update_conversation_endpoint(conversation_id: str, request: ConversationUpdate):
    """更新对话标题"""
    try:
        success = await db.update_conversation_title(conversation_id, request.title)
        if success:
            return {"message": "对话标题更新成功"}
        else:
            raise HTTPException(status_code=404, detail="对话不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新对话标题失败: {str(e)}")

@app.put("/api/conversations/{conversation_id}/star")
async def star_conversation_endpoint(conversation_id: str, request: ConversationStar):
    """更新对话收藏状态"""
    try:
        success = await db.update_conversation_starred(conversation_id, 1 if request.starred else 0)
        if success:
            return {"message": "对话收藏状态更新成功"}
        else:
            raise HTTPException(status_code=404, detail="对话不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新对话收藏状态失败: {str(e)}")

@app.get("/docs_manage", response_class=HTMLResponse)
async def docs_manage(request: Request):
    """文档管理页面"""
    return templates.TemplateResponse("docs.html", {"request": request})

@app.get("/mcp.html", response_class=HTMLResponse)
async def mcp_page(request: Request):
    """MCP管理页面"""
    return templates.TemplateResponse("mcp.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )