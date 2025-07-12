#!/usr/bin/env python3
"""系统监控脚本

用于监控Docker容器的运行状态、性能指标和系统健康状况
"""

import asyncio
import json
import time
import psutil
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.monitoring = False
        self.metrics_history = []
        self.max_history = 100  # 保留最近100条记录
    
    async def get_api_metrics(self) -> Dict[str, Any]:
        """获取API性能指标"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/documents/status") as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "response_time": response_time,
                            "api_data": data
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "response_time": response_time,
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # 网络IO
            net_io = psutil.net_io_counters()
            
            # 进程信息
            process_count = len(psutil.pids())
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                },
                "processes": process_count
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """获取应用程序特定指标"""
        try:
            metrics = {}
            
            # 检查存储目录大小
            storage_path = Path("storage")
            if storage_path.exists():
                storage_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
                metrics["storage_size"] = storage_size
            
            # 检查上传目录
            uploads_path = Path("uploads")
            if uploads_path.exists():
                upload_files = list(uploads_path.glob('*'))
                metrics["uploaded_files"] = len(upload_files)
                uploads_size = sum(f.stat().st_size for f in upload_files if f.is_file())
                metrics["uploads_size"] = uploads_size
            
            # 检查缓存目录
            cache_path = Path("cache")
            if cache_path.exists():
                cache_files = list(cache_path.glob('*'))
                metrics["cache_files"] = len(cache_files)
                cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
                metrics["cache_size"] = cache_size
            
            # 检查日志目录
            logs_path = Path("logs")
            if logs_path.exists():
                log_files = list(logs_path.rglob('*.log'))
                metrics["log_files"] = len(log_files)
                logs_size = sum(f.stat().st_size for f in log_files if f.is_file())
                metrics["logs_size"] = logs_size
            
            return metrics
        except Exception as e:
            return {"error": str(e)}
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """收集所有指标"""
        timestamp = datetime.now().isoformat()
        
        # 并行收集指标
        api_metrics_task = asyncio.create_task(self.get_api_metrics())
        
        system_metrics = self.get_system_metrics()
        app_metrics = self.get_application_metrics()
        api_metrics = await api_metrics_task
        
        metrics = {
            "timestamp": timestamp,
            "system": system_metrics,
            "application": app_metrics,
            "api": api_metrics
        }
        
        # 添加到历史记录
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        return metrics
    
    def format_bytes(self, bytes_value: int) -> str:
        """格式化字节数"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """打印格式化的指标"""
        print(f"\n{'='*60}")
        print(f"系统监控报告 - {metrics['timestamp']}")
        print(f"{'='*60}")
        
        # 系统指标
        if 'system' in metrics and 'error' not in metrics['system']:
            sys_metrics = metrics['system']
            print(f"\n🖥️  系统性能:")
            print(f"   CPU使用率: {sys_metrics['cpu']['percent']:.1f}% ({sys_metrics['cpu']['count']} 核心)")
            print(f"   内存使用: {sys_metrics['memory']['percent']:.1f}% ({self.format_bytes(sys_metrics['memory']['used'])}/{self.format_bytes(sys_metrics['memory']['total'])})")
            print(f"   磁盘使用: {sys_metrics['disk']['percent']:.1f}% ({self.format_bytes(sys_metrics['disk']['used'])}/{self.format_bytes(sys_metrics['disk']['total'])})")
            print(f"   进程数量: {sys_metrics['processes']}")
        
        # 应用指标
        if 'application' in metrics and 'error' not in metrics['application']:
            app_metrics = metrics['application']
            print(f"\n📁 应用数据:")
            if 'uploaded_files' in app_metrics:
                print(f"   上传文件: {app_metrics['uploaded_files']} 个 ({self.format_bytes(app_metrics.get('uploads_size', 0))})")
            if 'storage_size' in app_metrics:
                print(f"   存储大小: {self.format_bytes(app_metrics['storage_size'])}")
            if 'cache_files' in app_metrics:
                print(f"   缓存文件: {app_metrics['cache_files']} 个 ({self.format_bytes(app_metrics.get('cache_size', 0))})")
            if 'log_files' in app_metrics:
                print(f"   日志文件: {app_metrics['log_files']} 个 ({self.format_bytes(app_metrics.get('logs_size', 0))})")
        
        # API指标
        if 'api' in metrics:
            api_metrics = metrics['api']
            print(f"\n🌐 API状态:")
            status_emoji = "✅" if api_metrics['status'] == 'healthy' else "❌"
            print(f"   状态: {status_emoji} {api_metrics['status']}")
            if 'response_time' in api_metrics:
                print(f"   响应时间: {api_metrics['response_time']:.3f}s")
            if 'error' in api_metrics:
                print(f"   错误: {api_metrics['error']}")
    
    async def start_monitoring(self, interval: int = 30):
        """开始监控"""
        self.monitoring = True
        print(f"开始系统监控，间隔: {interval}秒")
        print("按 Ctrl+C 停止监控")
        
        try:
            while self.monitoring:
                metrics = await self.collect_metrics()
                self.print_metrics(metrics)
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
            self.monitoring = False
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
    
    def save_metrics_to_file(self, filename: str = None):
        """保存指标到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        
        print(f"指标已保存到: {filename}")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='系统监控工具')
    parser.add_argument('--interval', '-i', type=int, default=30, help='监控间隔（秒）')
    parser.add_argument('--once', action='store_true', help='只运行一次')
    parser.add_argument('--save', '-s', type=str, help='保存指标到文件')
    parser.add_argument('--url', default='http://localhost:8000', help='API基础URL')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.url)
    
    if args.once:
        # 只运行一次
        metrics = await monitor.collect_metrics()
        monitor.print_metrics(metrics)
        if args.save:
            monitor.save_metrics_to_file(args.save)
    else:
        # 持续监控
        try:
            await monitor.start_monitoring(args.interval)
        finally:
            if args.save:
                monitor.save_metrics_to_file(args.save)


if __name__ == "__main__":
    asyncio.run(main())