"""FastAPI 依赖注入"""

from __future__ import annotations

# Service 实例（懒加载单例）
_consultation_service = None
_session_service = None


def get_consultation_service():
    """获取 ConsultationService 实例"""
    global _consultation_service
    if _consultation_service is None:
        from app.service.consultation_service import ConsultationService
        _consultation_service = ConsultationService()
    return _consultation_service


def get_session_service():
    """获取 SessionService 实例"""
    global _session_service
    if _session_service is None:
        from app.service.session_service import SessionService
        _session_service = SessionService()
    return _session_service
