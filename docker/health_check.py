#!/usr/bin/env python3
"""健康检查脚本

用于监控系统运行状态，可以被Docker健康检查或外部监控系统调用
"""

import sys
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 10
    
    async def check_api_health(self) -> Dict[str, Any]:
        """检查API健康状态"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/documents/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "api_response": data,
                            "response_time": response.headers.get("X-Response-Time", "unknown")
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"API返回状态码: {response.status}",
                            "response_time": response.headers.get("X-Response-Time", "unknown")
                        }
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "error": "API请求超时"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"API请求失败: {str(e)}"
            }
    
    def check_file_system(self) -> Dict[str, Any]:
        """检查文件系统状态"""
        try:
            required_dirs = [
                "storage",
                "uploads", 
                "cache",
                "logs"
            ]
            
            missing_dirs = []
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                return {
                    "status": "warning",
                    "message": f"缺少目录: {', '.join(missing_dirs)}"
                }
            
            # 检查存储空间
            storage_path = Path("storage")
            if storage_path.exists():
                # 简单的磁盘空间检查
                import shutil
                total, used, free = shutil.disk_usage(storage_path)
                free_gb = free // (1024**3)
                
                if free_gb < 1:  # 少于1GB
                    return {
                        "status": "warning",
                        "message": f"磁盘空间不足: 剩余 {free_gb}GB"
                    }
            
            return {
                "status": "healthy",
                "message": "文件系统正常"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"文件系统检查失败: {str(e)}"
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """检查依赖项"""
        try:
            import torch
            import faiss
            import sentence_transformers
            import openai
            
            # 检查CUDA可用性
            cuda_available = torch.cuda.is_available()
            
            return {
                "status": "healthy",
                "cuda_available": cuda_available,
                "torch_version": torch.__version__,
                "dependencies": "正常"
            }
            
        except ImportError as e:
            return {
                "status": "unhealthy",
                "error": f"依赖项缺失: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"依赖项检查失败: {str(e)}"
            }
    
    async def run_full_check(self) -> Dict[str, Any]:
        """运行完整的健康检查"""
        results = {
            "timestamp": asyncio.get_event_loop().time(),
            "overall_status": "healthy"
        }
        
        # API健康检查
        api_result = await self.check_api_health()
        results["api"] = api_result
        
        # 文件系统检查
        fs_result = self.check_file_system()
        results["filesystem"] = fs_result
        
        # 依赖项检查
        deps_result = self.check_dependencies()
        results["dependencies"] = deps_result
        
        # 确定总体状态
        all_checks = [api_result, fs_result, deps_result]
        if any(check["status"] == "unhealthy" for check in all_checks):
            results["overall_status"] = "unhealthy"
        elif any(check["status"] == "warning" for check in all_checks):
            results["overall_status"] = "warning"
        
        return results


async def main():
    """主函数"""
    checker = HealthChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # 简单检查，仅检查API
        result = await checker.check_api_health()
        if result["status"] == "healthy":
            print("OK")
            sys.exit(0)
        else:
            print(f"FAIL: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        # 完整检查
        result = await checker.run_full_check()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result["overall_status"] == "healthy":
            sys.exit(0)
        elif result["overall_status"] == "warning":
            sys.exit(0)  # 警告不算失败
        else:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())