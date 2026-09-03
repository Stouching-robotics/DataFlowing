"""
传感器配置对话框 —— PyQt5 实现的矩阵选择和手掌配置界面。
"""

import json
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QRadioButton, QScrollArea, QWidget, QButtonGroup,
    QDialogButtonBox, QGridLayout,
)


# ═══════════════════════════════════════════════════════
#  通用矩阵行列选择对话框
# ═══════════════════════════════════════════════════════

class MatrixConfigDialog(QDialog):
    """通用矩阵行列配置 —— 按钮点击选择行/列序列。"""

    def __init__(self, parent, title: str, config: dict):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 480)

        self._rows = list(config.get("rows", []))
        self._cols = list(config.get("cols", []))
        self._order = config.get("axis_order", "row_col")

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 映射方向
        order_group = QGroupBox("映射渲染方向 (Axis Mapping)")
        order_layout = QHBoxLayout(order_group)
        bg = QButtonGroup(self)
        self._rb_rowcol = QRadioButton("先行后列 (横向列, 纵向行)")
        self._rb_colrow = QRadioButton("先列后行 (横向行, 纵向列)")
        bg.addButton(self._rb_rowcol)
        bg.addButton(self._rb_colrow)
        if self._order == "row_col":
            self._rb_rowcol.setChecked(True)
        else:
            self._rb_colrow.setChecked(True)
        order_layout.addWidget(self._rb_rowcol)
        order_layout.addWidget(self._rb_colrow)
        layout.addWidget(order_group)

        # 行列选择
        lists_layout = QHBoxLayout()

        # 行选择
        row_group = QGroupBox("行 (Rows 00-15)")
        row_vbox = QVBoxLayout(row_group)
        self._row_label = QLabel()
        self._row_label.setStyleSheet("color:#1565C0; font-weight:bold;")
        self._row_label.setWordWrap(True)
        row_vbox.addWidget(self._row_label)

        row_grid = QGridLayout()
        for i in range(16):
            btn = QPushButton(f"{i:02d}")
            btn.setFixedSize(36, 28)
            btn.clicked.connect(lambda checked, x=i: self._add_row(x))
            row_grid.addWidget(btn, i // 4, i % 4)
        row_vbox.addLayout(row_grid)

        row_btn_layout = QHBoxLayout()
        row_btn_layout.addWidget(QPushButton("撤销", clicked=self._undo_row))
        row_btn_layout.addWidget(QPushButton("清空", clicked=self._clear_row))
        row_vbox.addLayout(row_btn_layout)
        lists_layout.addWidget(row_group)

        # 列选择
        col_group = QGroupBox("列 (Cols 00-15)")
        col_vbox = QVBoxLayout(col_group)
        self._col_label = QLabel()
        self._col_label.setStyleSheet("color:#1565C0; font-weight:bold;")
        self._col_label.setWordWrap(True)
        col_vbox.addWidget(self._col_label)

        col_grid = QGridLayout()
        for i in range(16):
            btn = QPushButton(f"{i:02d}")
            btn.setFixedSize(36, 28)
            btn.clicked.connect(lambda checked, x=i: self._add_col(x))
            col_grid.addWidget(btn, i // 4, i % 4)
        col_vbox.addLayout(col_grid)

        col_btn_layout = QHBoxLayout()
        col_btn_layout.addWidget(QPushButton("撤销", clicked=self._undo_col))
        col_btn_layout.addWidget(QPushButton("清空", clicked=self._clear_col))
        col_vbox.addLayout(col_btn_layout)
        lists_layout.addWidget(col_group)

        layout.addLayout(lists_layout)

        # 确认按钮
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._update_labels()

    def _add_row(self, val):
        if val not in self._rows:
            self._rows.append(val)
            self._update_labels()

    def _undo_row(self):
        if self._rows:
            self._rows.pop()
            self._update_labels()

    def _clear_row(self):
        self._rows.clear()
        self._update_labels()

    def _add_col(self, val):
        if val not in self._cols:
            self._cols.append(val)
            self._update_labels()

    def _undo_col(self):
        if self._cols:
            self._cols.pop()
            self._update_labels()

    def _clear_col(self):
        self._cols.clear()
        self._update_labels()

    def _update_labels(self):
        r_str = " → ".join(map(str, self._rows)) if self._rows else "(无)"
        c_str = " → ".join(map(str, self._cols)) if self._cols else "(无)"
        self._row_label.setText(f"已选行:\n{r_str}")
        self._col_label.setText(f"已选列:\n{c_str}")

    def get_config(self) -> dict:
        order = "row_col" if self._rb_rowcol.isChecked() else "col_row"
        return {
            "rows": list(self._rows),
            "cols": list(self._cols),
            "axis_order": order,
        }


# ═══════════════════════════════════════════════════════
#  仿生手掌配置对话框
# ═══════════════════════════════════════════════════════

class HandConfigDialog(QDialog):
    """仿生手掌 —— 各手指区域的矩阵配置。"""

    def __init__(self, parent, config: dict):
        super().__init__(parent)
        self.setWindowTitle("仿生手掌映射配置")
        self.setMinimumSize(650, 600)
        self._config = {k: dict(v) for k, v in config.items()}
        self._labels = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("🦾 仿生手掌模式 - 矩阵配置"))
        layout.addWidget(QLabel('点击"选择"为每个部位配置传感器行列映射'))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)

        for key, item in self._config.items():
            name = item.get("name", key)
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 2, 0, 2)

            row_layout.addWidget(QLabel(name))

            label = QLabel(self._format_cfg(item))
            label.setStyleSheet("color:#1565C0;")
            label.setMinimumWidth(300)
            self._labels[key] = label
            row_layout.addWidget(label, 1)

            btn = QPushButton("选择")
            btn.clicked.connect(
                lambda checked, k=key: self._open_sub_config(k)
            )
            row_layout.addWidget(btn)

            container_layout.addWidget(row_widget)

        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _format_cfg(self, item):
        rows = item.get("rows", [])
        cols = item.get("cols", [])
        order = item.get("axis_order", "row_col")
        r_str = ",".join(map(str, rows)) if rows else "无"
        c_str = ",".join(map(str, cols)) if cols else "无"
        dir_str = "先行后列" if order == "row_col" else "先列后行"
        return f"[{dir_str}] 行:[{r_str}] 列:[{c_str}]"

    def _open_sub_config(self, key):
        item = self._config[key]
        sub_cfg = {
            "rows": list(item.get("rows", [])),
            "cols": list(item.get("cols", [])),
            "axis_order": item.get("axis_order", "row_col"),
        }
        dlg = MatrixConfigDialog(
            self, f"配置 - {item.get('name', key)}", sub_cfg
        )
        if dlg.exec_() == dlg.Accepted:
            new_cfg = dlg.get_config()
            item["rows"] = new_cfg["rows"]
            item["cols"] = new_cfg["cols"]
            item["axis_order"] = new_cfg["axis_order"]
            self._labels[key].setText(self._format_cfg(item))

    def get_config(self) -> dict:
        return self._config
