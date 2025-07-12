"""
리스크 관리 클래스
포지션 크기, 포트폴리오 리스크, Stop Loss/Take Profit 관리
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tools import st_utils as stu

logger = stu.create_logger()

class RiskManagerError(Exception):
    """리스크 관리 관련 예외"""
    pass

class RiskManager:
    """
    리스크 관리자
    
    주요 기능:
    - 포지션 크기 계산
    - 포트폴리오 리스크 모니터링
    - Stop Loss/Take Profit 관리
    - 일일 손실 한도 체크
    - 최대 보유 종목 수 제한
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 거래 설정 로드
        trading_config = config.get('trading', {})
        self.max_positions = trading_config.get('max_positions', 20)
        self.position_size_pct = trading_config.get('position_size_pct', 5.0)  # 계좌 대비 %
        self.stop_loss_pct = trading_config.get('stop_loss_pct', -5.0)
        self.take_profit_pct = trading_config.get('take_profit_pct', 10.0)
        self.daily_loss_limit_pct = trading_config.get('daily_loss_limit_pct', -10.0)
        
        # 포지션 추적
        self.positions = {}  # {code: position_info}
        self.daily_pnl = 0.0
        self.account_balance = 0.0
        
        logger.info(f"RiskManager 초기화 완료 (최대포지션: {self.max_positions}, 포지션크기: {self.position_size_pct}%)")
        
    def calculate_position_size(self, 
                               code: str, 
                               current_price: float, 
                               account_balance: float,
                               volatility: float = None) -> Dict[str, Any]:
        """
        포지션 크기 계산
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            account_balance: 계좌 잔고
            volatility: 변동성 (선택)
            
        Returns:
            Dict: 포지션 정보 (수량, 금액, 리스크 등)
        """
        try:
            # 기본 포지션 크기 계산
            base_amount = account_balance * (self.position_size_pct / 100)
            
            # 변동성 조정 (변동성이 높으면 포지션 크기 축소)
            if volatility:
                volatility_factor = min(1.0, 0.2 / max(volatility, 0.1))  # 20% 기준
                adjusted_amount = base_amount * volatility_factor
            else:
                adjusted_amount = base_amount
                
            # 수량 계산 (100주 단위로 반올림)
            quantity = int(adjusted_amount / current_price / 100) * 100
            actual_amount = quantity * current_price
            
            # Stop Loss/Take Profit 가격 계산
            stop_loss_price = current_price * (1 + self.stop_loss_pct / 100)
            take_profit_price = current_price * (1 + self.take_profit_pct / 100)
            
            position_info = {
                'code': code,
                'quantity': quantity,
                'entry_price': current_price,
                'amount': actual_amount,
                'stop_loss_price': round(stop_loss_price, 0),
                'take_profit_price': round(take_profit_price, 0),
                'max_loss': actual_amount * (self.stop_loss_pct / 100),
                'max_profit': actual_amount * (self.take_profit_pct / 100),
                'entry_time': datetime.now(),
                'volatility': volatility
            }
            
            logger.info(f"포지션 크기 계산 완료: {code}, 수량: {quantity}, 금액: {actual_amount:,.0f}원")
            return position_info
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 실패: {code}, {e}")
            raise RiskManagerError(f"포지션 크기 계산 실패: {e}")
            
    def check_portfolio_risk(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        포트폴리오 전체 리스크 체크
        
        Args:
            positions: 현재 포지션 정보
            
        Returns:
            Dict: 리스크 분석 결과
        """
        try:
            total_positions = len(positions)
            total_amount = sum(pos.get('amount', 0) for pos in positions.values())
            total_max_loss = sum(pos.get('max_loss', 0) for pos in positions.values())
            
            # 섹터별 집중도 분석 (추후 구현)
            sector_concentration = self._analyze_sector_concentration(positions)
            
            # 리스크 레벨 계산
            risk_level = self._calculate_risk_level(total_positions, total_amount, total_max_loss)
            
            risk_analysis = {
                'total_positions': total_positions,
                'max_positions': self.max_positions,
                'position_utilization': total_positions / self.max_positions,
                'total_amount': total_amount,
                'total_max_loss': total_max_loss,
                'daily_pnl': self.daily_pnl,
                'risk_level': risk_level,
                'sector_concentration': sector_concentration,
                'warnings': []
            }
            
            # 경고 체크
            if total_positions >= self.max_positions:
                risk_analysis['warnings'].append(f"최대 포지션 수 도달: {total_positions}/{self.max_positions}")
                
            if self.daily_pnl <= (self.account_balance * self.daily_loss_limit_pct / 100):
                risk_analysis['warnings'].append(f"일일 손실 한도 도달: {self.daily_pnl:,.0f}원")
                
            return risk_analysis
            
        except Exception as e:
            logger.error(f"포트폴리오 리스크 체크 실패: {e}")
            return {'error': str(e)}
            
    def check_stop_loss_take_profit(self, 
                                   code: str, 
                                   current_price: float, 
                                   position_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stop Loss/Take Profit 조건 체크
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            position_info: 포지션 정보
            
        Returns:
            Dict: 매도 신호 정보
        """
        try:
            entry_price = position_info.get('entry_price', 0)
            stop_loss_price = position_info.get('stop_loss_price', 0)
            take_profit_price = position_info.get('take_profit_price', 0)
            quantity = position_info.get('quantity', 0)
            
            # 수익률 계산
            return_pct = ((current_price - entry_price) / entry_price) * 100
            unrealized_pnl = (current_price - entry_price) * quantity
            
            signal = {
                'code': code,
                'current_price': current_price,
                'entry_price': entry_price,
                'return_pct': round(return_pct, 2),
                'unrealized_pnl': round(unrealized_pnl, 0),
                'action': 'hold',
                'reason': '',
                'urgency': 'normal'
            }
            
            # Stop Loss 체크
            if current_price <= stop_loss_price:
                signal.update({
                    'action': 'sell',
                    'reason': f'Stop Loss 발동 (목표: {stop_loss_price:,.0f}, 현재: {current_price:,.0f})',
                    'urgency': 'high'
                })
                
            # Take Profit 체크
            elif current_price >= take_profit_price:
                signal.update({
                    'action': 'sell',
                    'reason': f'Take Profit 발동 (목표: {take_profit_price:,.0f}, 현재: {current_price:,.0f})',
                    'urgency': 'normal'
                })
                
            return signal
            
        except Exception as e:
            logger.error(f"Stop Loss/Take Profit 체크 실패: {code}, {e}")
            return {'error': str(e)}
            
    def update_stop_loss_take_profit(self, 
                                   code: str, 
                                   new_stop_loss: float = None, 
                                   new_take_profit: float = None) -> bool:
        """
        Stop Loss/Take Profit 동적 업데이트
        
        Args:
            code: 종목 코드
            new_stop_loss: 새로운 Stop Loss 가격
            new_take_profit: 새로운 Take Profit 가격
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if code not in self.positions:
                logger.warning(f"포지션이 존재하지 않음: {code}")
                return False
                
            if new_stop_loss:
                self.positions[code]['stop_loss_price'] = new_stop_loss
                logger.info(f"Stop Loss 업데이트: {code}, {new_stop_loss:,.0f}원")
                
            if new_take_profit:
                self.positions[code]['take_profit_price'] = new_take_profit
                logger.info(f"Take Profit 업데이트: {code}, {new_take_profit:,.0f}원")
                
            return True
            
        except Exception as e:
            logger.error(f"Stop Loss/Take Profit 업데이트 실패: {code}, {e}")
            return False
            
    def add_position(self, position_info: Dict[str, Any]) -> bool:
        """포지션 추가"""
        try:
            code = position_info.get('code')
            if len(self.positions) >= self.max_positions:
                logger.warning(f"최대 포지션 수 초과: {len(self.positions)}/{self.max_positions}")
                return False
                
            self.positions[code] = position_info
            logger.info(f"포지션 추가: {code}")
            return True
            
        except Exception as e:
            logger.error(f"포지션 추가 실패: {e}")
            return False
            
    def remove_position(self, code: str) -> bool:
        """포지션 제거"""
        try:
            if code in self.positions:
                del self.positions[code]
                logger.info(f"포지션 제거: {code}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"포지션 제거 실패: {code}, {e}")
            return False
            
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """포트폴리오 요약 정보"""
        try:
            total_amount = sum(pos.get('amount', 0) for pos in self.positions.values())
            total_max_loss = sum(pos.get('max_loss', 0) for pos in self.positions.values())
            total_max_profit = sum(pos.get('max_profit', 0) for pos in self.positions.values())
            
            return {
                'position_count': len(self.positions),
                'max_positions': self.max_positions,
                'total_amount': total_amount,
                'total_max_loss': total_max_loss,
                'total_max_profit': total_max_profit,
                'daily_pnl': self.daily_pnl,
                'account_balance': self.account_balance,
                'positions': list(self.positions.keys())
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 요약 생성 실패: {e}")
            return {}
            
    def _analyze_sector_concentration(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """섹터별 집중도 분석 (추후 구현)"""
        # TODO: 종목별 섹터 정보를 활용한 집중도 분석
        return {}
        
    def _calculate_risk_level(self, total_positions: int, total_amount: float, total_max_loss: float) -> str:
        """리스크 레벨 계산"""
        if total_positions == 0:
            return 'none'
            
        position_ratio = total_positions / self.max_positions
        loss_ratio = abs(total_max_loss) / max(self.account_balance, 1)
        
        if position_ratio > 0.8 or loss_ratio > 0.15:
            return 'high'
        elif position_ratio > 0.5 or loss_ratio > 0.10:
            return 'medium'
        else:
            return 'low' 