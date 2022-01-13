# -*-coding:utf-8-*-
from PyQt5.QtWidgets import QLabel, QLineEdit, QApplication, QPushButton
from PyQt5.QtGui import *
# import os


# from PyQt5.QtCore import *


def pic(page):
    label_monthly_returns = QLabel()
    label_monthly_returns.setPixmap(QPixmap('../frontend/monthly_returns.png').scaled(500, 400))
    # label_monthly_returns.setEnabled(False)
    page.grid.addWidget(label_monthly_returns, 0, 0)

    label_pnl = QLabel()
    label_pnl.setPixmap(QPixmap('../frontend/daily_PnL.png').scaled(500, 400))
    # # label_pnl.setEnabled(False)
    page.grid.addWidget(label_pnl, 1, 0)

    label_content = QLabel()
    label_content.setFont(page.font_content)
    label_content.setText("Final Sharpe Ratio")
    page.grid.addWidget(label_content, 0, 1)
    page.line_dr = QLineEdit(page.widget)
    page.line_dr.setEnabled(False)
    page.line_dr.setStyleSheet("QLineEdit{background-color:white;color:black}")
    page.grid.addWidget(page.line_dr, 0, 2)

    # Final PnL
    label_final_pnl = QLabel()
    label_final_pnl.setFont(page.font_content)
    label_final_pnl.setText("Final PnL")
    page.grid.addWidget(label_final_pnl, 1, 1)
    page.line_final_pnl = QLineEdit(page.widget)
    page.line_final_pnl.setEnabled(False)
    page.line_final_pnl.setStyleSheet("QLineEdit{background-color:white;color:black}")
    page.grid.addWidget(page.line_final_pnl, 1, 2)

    # Final PnL
    label_mdd = QLabel()
    label_mdd.setFont(page.font_content)
    label_mdd.setText("Max Drawdown(Monthly Returns)")
    page.grid.addWidget(label_mdd, 3, 1)
    page.line_mdd = QLineEdit(page.widget)
    page.line_mdd.setEnabled(False)
    page.line_mdd.setStyleSheet("QLineEdit{background-color:white;color:black}")
    page.grid.addWidget(page.line_mdd, 3, 2)

def result(page):
    # 标题
    # label_main = QLabel()
    # label_main.setFont(page.font_main)
    # label_main.setText("  Back-test Results")
    # page.grid.addWidget(label_main, 1, 0, 1, 4)
    # # 文字
    # label_hint = QLabel()
    # label_hint.setFont(page.font_content)
    # label_hint.setText("      These are the results")
    # page.grid.addWidget

    # QApplication.processEvents()
    #pics
    # F.fig.add_subplot(2, 1, 0)
    # label_monthly_returns = QLabel()
    # label_monthly_returns.setPixmap(QPixmap('../frontend/monthly_returns.png').scaled(500, 400))
    # # label_monthly_returns.setEnabled(False)
    # page.grid.addWidget(label_monthly_returns, 0, 0)
    #
    # label_pnl = QLabel()
    # label_pnl.setPixmap(QPixmap('../frontend/daily_PnL.png').scaled(500, 400))
    # # # label_pnl.setEnabled(False)
    # page.grid.addWidget(label_pnl, 1, 0)
    # page.pic_monthly_returns = QLineEdit(page.widget)
    # page.pic_monthly_returns.setEnabled(False)
    # page.grid.addWidget(page.pic_monthly_returns, 1, 1)

    # Sharpe Ratio
    label_content = QLabel()
    label_content.setFont(page.font_content)
    label_content.setText("Final Sharpe Ratio")
    page.grid.addWidget(label_content, 0, 1)
    page.line_dr = QLineEdit(page.widget)
    page.line_dr.setEnabled(False)
    page.line_dr.setStyleSheet("QLineEdit{background-color:white;color:black}")
    page.grid.addWidget(page.line_dr, 0, 2)

    # Final PnL
    label_final_pnl = QLabel()
    label_final_pnl.setFont(page.font_content)
    label_final_pnl.setText("Final PnL")
    page.grid.addWidget(label_final_pnl, 1, 1)
    page.line_final_pnl = QLineEdit(page.widget)
    page.line_final_pnl.setEnabled(False)
    page.line_final_pnl.setStyleSheet("QLineEdit{background-color:white;color:black}")
    page.grid.addWidget(page.line_final_pnl, 1, 2)

    # Final PnL
    label_mdd = QLabel()
    label_mdd.setFont(page.font_content)
    label_mdd.setText("Max Drawdown(Monthly Returns)")
    page.grid.addWidget(label_mdd, 3, 1)
    page.line_mdd = QLineEdit(page.widget)
    page.line_mdd.setEnabled(False)
    page.line_mdd.setStyleSheet("QLineEdit{background-color:white;color:black}")
    page.grid.addWidget(page.line_mdd, 3, 2)

    # QApplication.processEvents()

    # btn_update = QPushButton('Update')
    # page.grid.addWidget(btn_update, 4, 0, 1, 4)
    # btn_update.setStyleSheet('''
    #         QPushButton:hover{color:red}
    #         QPushButton{font-size:18px;
    #                     font-weight:200;
    #         }''')
    # btn_update.clicked.connect(pic.show())

    # # 二叉树价格
    # label_bt = QLabel()
    # label_bt.setFont(page.font_content)
    # label_bt.setText("      Binary Tree Model Price")
    # page.grid.addWidget(label_bt, 4, 0)
    # page.line_bt = QLineEdit(page.widget)
    # page.line_bt.setEnabled(False)
    # page.line_bt.setStyleSheet("QLineEdit{background-color:white;color:black}")
    # page.grid.addWidget(page.line_bt, 4, 1)