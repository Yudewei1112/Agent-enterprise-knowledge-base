from fastapi import APIRouter, HTTPException
import uuid
import json
from datetime import datetime
from fastmcp import Client
from fastmcp.client.transports import (PythonStdioTransport, SSETransport)
from database import db

router = APIRouter(prefix="/api/mcp", tags=["mcp"])

# Function to fetch tools from an MCP server
# 参考mcp定义：https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-03-26/server/tools.mdx
async def fetch_mcp_tools(server_url: str, auth_type: str, auth_value: str) -> list:
    try:
        async with Client(SSETransport(server_url)) as client:         
            tools = await client.list_tools()
            print(tools)
        # Ensure tools have required fields
        return [
            {
                "id": str(uuid.uuid4()),
                "name": tool.name,
                "description": tool.description,
                "input_schema": json.dumps(tool.inputSchema)
            }
            for tool in tools
        ]
    except Exception as e:
        print(f"Error fetching tools from {server_url}: {str(e)}")
        return []

# Create MCP server
@router.post("/servers")
async def create_mcp_server(server: dict):
    try:
        server_id = await db.add_mcp_server(
            server["name"],
            server["url"],
            server.get("description", ""),
            server.get("auth_type", "none"),
            server.get("auth_value", "")
        )
        
        # Fetch and store tools
        tools = await fetch_mcp_tools(server["url"], server.get("auth_type", "none"), server.get("auth_value", ""))
        for tool in tools:
            await db.add_mcp_tool(
                server_id,
                tool["name"],
                tool["description"],
                json.loads(tool["input_schema"]) if isinstance(tool["input_schema"], str) else tool["input_schema"]
            )
        
        return {"id": server_id, "message": "MCP server created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create MCP server: {str(e)}")

# List MCP servers
@router.get("/servers")
async def list_mcp_servers():
    try:
        servers = await db.get_mcp_servers()
        return servers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list MCP servers: {str(e)}")

# Get specific MCP server
@router.get("/servers/{server_id}")
async def get_mcp_server(server_id: str):
    try:
        server = await db.get_mcp_server(server_id)
        if not server:
            raise HTTPException(status_code=404, detail="MCP server not found")
        return server
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get MCP server: {str(e)}")

# Update MCP server
@router.put("/servers/{server_id}")
async def update_mcp_server(server_id: str, server: dict):
    try:
        # Update server details
        success = await db.update_mcp_server(
            server_id,
            server["name"],
            server["url"],
            server.get("description", ""),
            server.get("auth_type", "none"),
            server.get("auth_value", "")
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="MCP server not found")
        
        # Delete existing tools for this server
        await db.delete_mcp_tools_by_server(server_id)
        
        # Fetch and store new tools
        tools = await fetch_mcp_tools(server["url"], server.get("auth_type", "none"), server.get("auth_value", ""))
        for tool in tools:
            await db.add_mcp_tool(
                server_id,
                tool["name"],
                tool["description"],
                json.loads(tool["input_schema"]) if isinstance(tool["input_schema"], str) else tool["input_schema"]
            )
        
        return {"message": "MCP server updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update MCP server: {str(e)}")

# Delete MCP server
@router.delete("/servers/{server_id}")
async def delete_mcp_server(server_id: str):
    try:
        # Delete associated tools first
        await db.delete_mcp_tools_by_server(server_id)
        
        # Delete server
        success = await db.delete_mcp_server(server_id)
        if not success:
            raise HTTPException(status_code=404, detail="MCP server not found")
        
        return {"message": "MCP server deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete MCP server: {str(e)}")

# Refresh tools for an MCP server
@router.post("/servers/{server_id}/refresh-tools")
async def refresh_mcp_server_tools(server_id: str):
    try:
        # Get server details
        server = await db.get_mcp_server(server_id)
        if not server:
            raise HTTPException(status_code=404, detail="MCP server not found")
        
        # Delete existing tools for this server
        await db.delete_mcp_tools_by_server(server_id)
        
        # Fetch and store new tools
        tools = await fetch_mcp_tools(server["url"], server["auth_type"], server["auth_value"])
        for tool in tools:
            await db.add_mcp_tool(
                server_id,
                tool["name"],
                tool["description"],
                json.loads(tool["input_schema"]) if isinstance(tool["input_schema"], str) else tool["input_schema"]
            )
        
        return {"message": "Tools refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh tools: {str(e)}")

# List tools (optionally filtered by server_id)
@router.get("/tools")
async def list_tools(server_id: str = None):
    try:
        tools = await db.get_mcp_tools()
        if server_id:
            tools = [tool for tool in tools if tool.get('server_id') == server_id]
        
        # 转换字段名以匹配前端期望的格式
        formatted_tools = []
        for tool in tools:
            formatted_tool = {
                'id': tool.get('tool_id'),
                'name': tool.get('tool_name'),
                'description': tool.get('tool_description'),
                'server_id': tool.get('server_id'),
                'server_name': tool.get('server_name'),
                'server_url': tool.get('server_url'),
                'input_schema': tool.get('input_schema')
            }
            formatted_tools.append(formatted_tool)
        
        return formatted_tools
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")

# Helper function to get MCP server details (used by process_stream_request)
async def get_mcp_server_details(server_id: str) -> dict:
    try:
        server = await db.get_mcp_server(server_id)
        if not server:
            raise HTTPException(status_code=404, detail="MCP server not found")
        return server
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get MCP server: {str(e)}")