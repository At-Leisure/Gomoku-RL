import inspect
import sys
import pathlib
import warnings
from pprint import pprint
from functools import wraps, partial
import typing
from typing import final, Literal, Mapping, TypeVar, Any, Generic, Iterable
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QFrame, QGroupBox, QLabel, QFormLayout, QSpinBox, QDoubleSpinBox, QTextBrowser,
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QCheckBox,
    QToolButton, QComboBox, QMenu, QWidgetAction)
from PyQt6.QtGui import QCursor


class EditorABC:

    @property
    def current_param(self): ...

    @current_param.setter
    def current_param(self, v): ...


class SelctorBase(QFrame, EditorABC):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.editer = QLineEdit(self)
        self.selector = QToolButton(self)
        self.selector.setToolTip('无可触发的按钮事件')

        try:
            self.selector.setText(r'🔗')
        except:
            self.selector.setText(r'&&')

        layout = QHBoxLayout()
        layout.addWidget(self.editer)
        layout.addWidget(self.selector)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self.selector.clicked.connect(self._select)

    def _select(self):
        fn, _ = QFileDialog.getSaveFileName()
        if fn:
            self.editer.setText(fn)

    @property
    def current_param(self):
        return self.editer.text()

    @current_param.setter
    def current_param(self, v):
        self.editer.setText(str(v))


class FileSaver(SelctorBase):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.selector.setToolTip('从文件对话框选择')

    def _select(self):
        fn, _ = QFileDialog.getSaveFileName()
        if fn:
            self.editer.setText(fn)


class FileGetter(FileSaver):

    def _select(self):
        fn, _ = QFileDialog.getOpenFileName()
        if fn:
            self.editer.setText(fn)


class IntEditor(QSpinBox, EditorABC):

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setMinimum(-2147483647)
        self.setMaximum(2147483647)

    @property
    def current_param(self):
        return self.value()

    @current_param.setter
    def current_param(self, v):
        self.setValue(v)


class UIntEditor(QSpinBox, EditorABC):

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setMinimum(0)
        self.setMaximum(2147483647)

    @property
    def current_param(self):
        return self.value()

    @current_param.setter
    def current_param(self, v):
        self.setValue(v)


class FloatEditor(QDoubleSpinBox, EditorABC):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setMinimum(-2147483647)
        self.setMaximum(2147483647)

    @property
    def current_param(self):
        return self.value()

    @current_param.setter
    def current_param(self, v):
        self.setValue(v)


class TextEditor(QLineEdit, EditorABC):
    @property
    def current_param(self):
        return self.text()

    @current_param.setter
    def current_param(self, v):
        self.setText(str(v))


class BoolEditor(QCheckBox, EditorABC):

    def __init__(self, parent):
        super().__init__(parent)
        self.stateChanged.connect(self._set_state_text)
        self._set_state_text()

    @property
    def current_param(self):
        return self.isChecked()

    @current_param.setter
    def current_param(self, v: bool):
        self.setChecked(v)

    def _set_state_text(self):
        self.setText('True' if self.isChecked() else 'False')
        if self.isChecked():
            self.setStyleSheet("QCheckBox{color:#000000}")
        else:
            self.setStyleSheet("QCheckBox{color:#aaaaaa}")


class ComboEditor(SelctorBase):

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.selector.setText('⮛')
        self.selector.setToolTip('点击触发选择菜单')
        self.menu = QMenu()
        self.cur = QCursor()

    def _select(self):
        self.menu.move(self.cur.pos())
        self.menu.show()

    @property
    def current_param(self):
        return self.editer.text()

    @current_param.setter
    def current_param(self, v: str | list[str] | tuple[str] | set[str]):
        """ 设置当前内容，或是设置可选的文本内容 """
        # 设置显示内容
        if isinstance(v, str):
            self.editer.setText(v)
        # 设置可选内容
        elif isinstance(v, list | tuple | set):
            if not all(isinstance(s, str) for s in v):
                warnings.warn('错误的类型')
                return
            self.menu.clear()
            for s in v:
                act = QWidgetAction(self)
                act.setText(s)
                act.triggered.connect(partial(self.editer.setText, s))
                self.menu.addAction(act)

        else:
            warnings.warn('错误的类型')


class SeparateLine(QLabel, EditorABC):

    @property
    def current_param(self):
        return SeparateLine

    @current_param.setter
    def current_param(self, v):
        self.setText(v)
