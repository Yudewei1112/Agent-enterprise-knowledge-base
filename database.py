import sqlite3
from datetime import datetime
import json
import uuid
import aiosqlite
import asyncio
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager
from config import config

class Database:
    def __init__(self, db_name=None):
        self.db_name = db_name if db_name else config.system_config["database_file"]
        self._init_db_sync()  # 同步初始化数据库
        self._connection_pool = asyncio.Queue(maxsize=5)  # 创建连接池
        self._initialized = False

    def _init_db_sync(self):
        """同步初始化数据库表"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            
            # 创建对话历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    starred INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            ''')
            # Create MCP servers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                description TEXT,
                auth_type TEXT,
                auth_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create MCP tools table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mcp_tools (
                id TEXT PRIMARY KEY,
                server_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                input_schema TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (server_id) REFERENCES mcp_servers(id)
            )
            ''')

            
            conn.commit()

    async def _initialize_pool(self):
        """初始化连接池"""
        if not self._initialized:
            for _ in range(5):  # 创建5个连接
                conn = await aiosqlite.connect(self.db_name)
                await self._connection_pool.put(conn)
            self._initialized = True

    @asynccontextmanager
    async def get_connection(self):
        """获取数据库连接的上下文管理器"""
        if not self._initialized:
            await self._initialize_pool()
        
        conn = await self._connection_pool.get()
        try:
            yield conn
        finally:
            await self._connection_pool.put(conn)

    async def create_conversation(self, title: str) -> str:
        """异步创建新的对话"""
        conversation_id = str(uuid.uuid4())
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO conversations (conversation_id, title, starred) VALUES (?, ?, 0)",
                (conversation_id, title)
            )
            await conn.commit()
            return conversation_id

    async def add_message(self, conversation_id: str, content: str, message_type: str) -> int:
        """异步添加新消息"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO messages (conversation_id, content, type) VALUES (?, ?, ?)",
                (conversation_id, content, message_type)
            )
            # 更新对话的更新时间
            await cursor.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                (conversation_id,)
            )
            await conn.commit()
            return cursor.lastrowid

    async def get_conversations(self, limit: int = 50) -> List[Tuple]:
        """异步获取最近的对话列表"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
                WITH FirstMessages AS (
                    SELECT 
                        conversation_id,
                        MIN(created_at) as first_message_time
                    FROM messages
                    WHERE type = 'user'
                    GROUP BY conversation_id
                )
                SELECT 
                    c.conversation_id,
                    c.title,
                    c.starred,
                    c.created_at,
                    c.updated_at,
                    GROUP_CONCAT(m.content || '|' || m.type || '|' || m.created_at, '||') as messages
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                LEFT JOIN FirstMessages fm ON c.conversation_id = fm.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.starred DESC, c.updated_at DESC
                LIMIT ?
            ''', (limit,))
            return await cursor.fetchall()

    async def get_messages(self, conversation_id: str) -> List[Tuple]:
        """异步获取指定对话的所有消息"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
                SELECT id, content, type, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            ''', (conversation_id,))
            return await cursor.fetchall()

    async def delete_conversation(self, conversation_id: str) -> bool:
        """异步删除对话及其所有消息"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                # 首先删除相关的消息
                await cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                # 然后删除对话
                await cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
                await conn.commit()
                return True
        except Exception as e:
            print(f"删除对话失败: {str(e)}")
            return False

    async def search_conversations(self, keyword: str) -> List[Tuple]:
        """异步搜索对话"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
                SELECT DISTINCT c.conversation_id, c.title, c.created_at, c.updated_at,
                       GROUP_CONCAT(m.content || '|' || m.type || '|' || m.created_at, '||') as messages
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                WHERE c.title LIKE ? OR m.content LIKE ?
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
            ''', (f'%{keyword}%', f'%{keyword}%'))
            return await cursor.fetchall()

    async def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """异步更新会话标题"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute(
                    "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                    (new_title, conversation_id)
                )
                await conn.commit()
                return True
        except Exception as e:
            print(f"更新会话标题失败: {str(e)}")
            return False

    async def update_conversation_starred(self, conversation_id: str, starred: int) -> bool:
        """异步更新对话的收藏状态"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute(
                    "UPDATE conversations SET starred = ? WHERE conversation_id = ?",
                    (starred, conversation_id)
                )
                await conn.commit()
                return True
        except Exception as e:
            print(f"更新对话收藏状态失败: {str(e)}")
            return False

    async def close(self):
        """关闭所有数据库连接"""
        try:
            if self._initialized:
                while not self._connection_pool.empty():
                    conn = await self._connection_pool.get()
                    await conn.close()
                self._initialized = False
                print("数据库连接池已关闭")
        except Exception as e:
            print(f"关闭数据库连接池时出现异常: {str(e)}")

    async def get_conversation_history(self, conversation_id: str, max_tokens: int = 100000) -> Tuple[List[Tuple], bool]:
        """
        获取对话历史记录
        参数:
            conversation_id: 对话ID
            max_tokens: 最大token数（默认100k）
        返回:
            Tuple[List[Tuple], bool]: (消息列表, 是否超出限制)
        """
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            # 获取所有消息，按时间排序
            await cursor.execute('''
                SELECT content, type, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            ''', (conversation_id,))
            messages = await cursor.fetchall()
            
            # 计算总字符数
            total_chars = sum(len(msg[0]) for msg in messages)
            
            # 如果超过限制，返回True表示超出限制
            if total_chars > max_tokens:
                return messages, True
            
            return messages, False

    async def get_conversation_context(self, conversation_id: str, max_tokens: int = 100000) -> Tuple[str, bool]:
        """
        获取对话上下文
        参数:
            conversation_id: 对话ID
            max_tokens: 最大token数（默认100k）
        返回:
            Tuple[str, bool]: (上下文字符串, 是否超出限制)
        """
        messages, is_overflow = await self.get_conversation_history(conversation_id, max_tokens)
        
        if is_overflow:
            return "", True
        
        # 构建上下文字符串
        context = []
        for content, msg_type, _ in messages:
            if msg_type == "user":
                context.append(f"用户: {content}")
            else:
                context.append(f"助手: {content}")
        
        return "\n".join(context), False

    async def get_mcp_tools(self) -> List[dict]:
        """获取所有MCP工具"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
                SELECT 
                    t.server_id,
                    s.name as server_name,
                    s.url as server_url,
                    t.id as tool_id,
                    t.name as tool_name,
                    t.description as tool_description,
                    t.input_schema
                FROM mcp_tools t
                JOIN mcp_servers s ON t.server_id = s.id
                ORDER BY s.name, t.name
            ''')
            rows = await cursor.fetchall()
            
            tools = []
            for row in rows:
                tool = {
                    'server_id': row[0],
                    'server_name': row[1],
                    'server_url': row[2],
                    'tool_id': row[3],
                    'tool_name': row[4],
                    'tool_description': row[5],
                    'input_schema': json.loads(row[6]) if row[6] else None
                }
                tools.append(tool)
            
            return tools
    
    async def add_mcp_server(self, name: str, url: str, description: str = None, 
                            auth_type: str = "none", auth_value: str = "") -> str:
        """添加MCP服务器，返回服务器ID"""
        try:
            server_id = str(uuid.uuid4())
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute(
                    "INSERT INTO mcp_servers (id, name, url, description, auth_type, auth_value, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                    (server_id, name, url, description, auth_type, auth_value)
                )
                await conn.commit()
                return server_id
        except Exception as e:
            print(f"添加MCP服务器失败: {str(e)}")
            raise e
    
    async def add_mcp_tool(self, server_id: str, name: str, description: str = None, 
                          input_schema: dict = None) -> bool:
        """添加MCP工具"""
        try:
            tool_id = str(uuid.uuid4())
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute(
                    "INSERT INTO mcp_tools (id, server_id, name, description, input_schema, created_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (tool_id, server_id, name, description, json.dumps(input_schema) if input_schema else None)
                )
                await conn.commit()
                return True
        except Exception as e:
            print(f"添加MCP工具失败: {str(e)}")
            return False
    
    async def get_mcp_servers(self) -> List[dict]:
        """获取所有MCP服务器"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "SELECT id, name, url, description, auth_type, auth_value, created_at, updated_at FROM mcp_servers"
            )
            rows = await cursor.fetchall()
            servers = []
            for row in rows:
                server = {
                    'id': row[0],
                    'name': row[1],
                    'url': row[2],
                    'description': row[3],
                    'auth_type': row[4],
                    'auth_value': row[5],
                    'created_at': row[6],
                    'updated_at': row[7]
                }
                servers.append(server)
            return servers
    
    async def get_mcp_server(self, server_id: str) -> Optional[dict]:
        """获取指定MCP服务器"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "SELECT id, name, url, description, auth_type, auth_value, created_at, updated_at FROM mcp_servers WHERE id = ?",
                (server_id,)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'url': row[2],
                    'description': row[3],
                    'auth_type': row[4],
                    'auth_value': row[5],
                    'created_at': row[6],
                    'updated_at': row[7]
                }
            return None
    
    async def update_mcp_server(self, server_id: str, name: str, url: str, 
                               description: str = None, auth_type: str = None, auth_value: str = None) -> bool:
        """更新MCP服务器"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute(
                    "UPDATE mcp_servers SET name = ?, url = ?, description = ?, auth_type = ?, auth_value = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (name, url, description, auth_type, auth_value, server_id)
                )
                await conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"更新MCP服务器失败: {str(e)}")
            return False
    
    async def delete_mcp_tools_by_server(self, server_id: str) -> bool:
        """删除指定服务器的所有工具"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute("DELETE FROM mcp_tools WHERE server_id = ?", (server_id,))
                await conn.commit()
                return True
        except Exception as e:
            print(f"删除MCP工具失败: {str(e)}")
            return False
    
    async def delete_mcp_server(self, server_id: str) -> bool:
        """删除MCP服务器"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
                await conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"删除MCP服务器失败: {str(e)}")
            return False

# 创建数据库实例
db = Database()

# 便捷函数，保持与原有代码的兼容性
def get_mcp_tools() -> List[dict]:
    """同步获取MCP工具列表（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，无法同步等待，返回空列表
            print("警告：在运行的事件循环中调用同步函数 get_mcp_tools，返回空列表")
            return []
        else:
            return loop.run_until_complete(db.get_mcp_tools())
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        return asyncio.run(db.get_mcp_tools())

def get_conversation_history(conversation_id: str, limit: int = None) -> List[dict]:
    """同步获取对话历史（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return []
        else:
            messages = loop.run_until_complete(db.get_messages(conversation_id))
            # 转换格式以保持兼容性
            result = []
            for msg in messages:
                result.append({
                    'role': 'user' if msg[2] == 'user' else 'assistant',
                    'content': msg[1],
                    'timestamp': msg[3]
                })
            return result
    except RuntimeError:
        messages = asyncio.run(db.get_messages(conversation_id))
        result = []
        for msg in messages:
            result.append({
                'role': 'user' if msg[2] == 'user' else 'assistant',
                'content': msg[1],
                'timestamp': msg[3]
            })
        return result

def save_message(conversation_id: str, role: str, content: str, metadata: dict = None) -> bool:
    """同步保存消息（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return True
        else:
            loop.run_until_complete(db.add_message(conversation_id, content, role))
            return True
    except RuntimeError:
        asyncio.run(db.add_message(conversation_id, content, role))
        return True
    except Exception:
        return False

def create_conversation(conversation_id: str, title: str) -> bool:
    """同步创建对话（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return True
        else:
            # 注意：原有的create_conversation会生成新ID，这里需要适配
            loop.run_until_complete(db.create_conversation(title))
            return True
    except RuntimeError:
        asyncio.run(db.create_conversation(title))
        return True
    except Exception:
        return False

def get_all_conversations() -> List[dict]:
    """同步获取所有对话（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return []
        else:
            conversations = loop.run_until_complete(db.get_conversations())
            # 转换格式以保持兼容性
            result = []
            for conv in conversations:
                result.append({
                    'id': conv[0],
                    'title': conv[1],
                    'is_starred': conv[2],
                    'created_at': conv[3],
                    'updated_at': conv[4]
                })
            return result
    except RuntimeError:
        conversations = asyncio.run(db.get_conversations())
        result = []
        for conv in conversations:
            result.append({
                'id': conv[0],
                'title': conv[1],
                'is_starred': conv[2],
                'created_at': conv[3],
                'updated_at': conv[4]
            })
        return result

def delete_conversation(conversation_id: str) -> bool:
    """同步删除对话（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return True
        else:
            return loop.run_until_complete(db.delete_conversation(conversation_id))
    except RuntimeError:
        return asyncio.run(db.delete_conversation(conversation_id))
    except Exception:
        return False

def update_conversation_title(conversation_id: str, title: str) -> bool:
    """同步更新对话标题（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return True
        else:
            return loop.run_until_complete(db.update_conversation_title(conversation_id, title))
    except RuntimeError:
        return asyncio.run(db.update_conversation_title(conversation_id, title))
    except Exception:
        return False

def update_conversation_star(conversation_id: str, is_starred: bool) -> bool:
    """同步更新对话收藏状态（用于兼容性）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return True
        else:
            return loop.run_until_complete(db.update_conversation_starred(conversation_id, 1 if is_starred else 0))
    except RuntimeError:
        return asyncio.run(db.update_conversation_starred(conversation_id, 1 if is_starred else 0))
    except Exception:
        return False