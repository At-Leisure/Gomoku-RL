""" 命令操作组 

>>> c = Group()
    c.register(测试)
    c.mainloop()"""

import inspect
import sys
import warnings
from pprint import pprint
from functools import wraps, partial
import typing
from typing import final, Literal, Mapping, TypeVar
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QFrame, QGroupBox, QLabel, QFormLayout, QSpinBox, QDoubleSpinBox, QTextBrowser,
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog)


class CmdBase:
    """ base """


class PathEditer(QFrame):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.editer = QLineEdit(self)
        self.selector = QPushButton(self)

        layout = QHBoxLayout()
        layout.addWidget(self.editer)
        layout.addWidget(self.selector)
        self.setLayout(layout)

        self.selector.clicked.connect(self._select)

    def _select(self):
        fn, _ = QFileDialog.getOpenFileName()
        if fn:
            self.editer.setText(fn)

    @property
    def current_param(self):
        return self.editer.text()

    @current_param.setter
    def current_param(self, v):
        self.editer.setText(str(v))


class IntEditer(QSpinBox):

    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.setMinimum(-2147483647)
        self.setMaximum(2147483647)

    @property
    def current_param(self):
        return self.value()

    @current_param.setter
    def current_param(self, v):
        self.setValue(v)


class UIntEditer(QSpinBox):

    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.setMinimum(0)
        self.setMaximum(2147483647)

    @property
    def current_param(self):
        return self.value()

    @current_param.setter
    def current_param(self, v):
        self.setValue(v)


class FloatEditer(QDoubleSpinBox):
    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.setMinimum(-2147483647)
        self.setMaximum(2147483647)

    @property
    def current_param(self):
        return self.value()

    @current_param.setter
    def current_param(self, v):
        self.setValue(v)


class TextEditer(QLineEdit):
    @property
    def current_param(self):
        return self.text()

    @current_param.setter
    def current_param(self, v):
        self.setText(str(v))


InputWidget = IntEditer | UIntEditer | PathEditer | FloatEditer | TextEditer


class ParamBox(QGroupBox):

    def __init__(self, parent, info: Mapping[str, str | type | list[Mapping[str, InputWidget | str]]] = ...):
        super().__init__('参数', parent)
        self.params_widgets = []

        layout = QFormLayout()
        for p in info['params']:
            w: InputWidget = p['annotation'](self)
            w.current_param = p['default']
            self.params_widgets.append(w)
            layout.addRow(p['name'], w)
        self.setLayout(layout)


class FuncFrame(QFrame):

    def __init__(self, parent: QWidget, info: Mapping[str, str | type | list[Mapping[str, InputWidget | str]]] = ...) -> None:
        super().__init__(parent)
        self.info = info
        param_frame = self
        desc_box = QGroupBox(self)
        desc_box.setTitle('描述')
        layout = QVBoxLayout()
        button = QPushButton()
        button.setText('执行')
        button.clicked.connect(self._run)
        layout.addWidget(button)
        label = QTextBrowser()
        # label.setText(info['doc'])
        label.setMarkdown(info['doc'])
        # label.setWordWrap(True)#启用断行
        layout.addWidget(label)
        desc_box.setLayout(layout)

        self.param_box = ParamBox(self, info)

        layout = QHBoxLayout()
        layout.addWidget(desc_box)
        layout.addWidget(self.param_box)
        param_frame.setLayout(layout)

    def _run(self):
        params = [w.current_param for w in self.param_box.params_widgets]
        self.info['func'](*params)


class Group:
    """ >>> # data template
    [
        {
            'cmds': [
                {
                    'doc': ' func ',
                    'func': <staticmethod(<function Test.func at 0x000001FEF5E6CCC0>)>,
                    'params': [
                        {
                            'annotation': <class '__main__.Int'>,
                            'default': <class 'inspect._empty'>,
                            'name': 'aa'
                        },
                        {
                            'annotation': <class '__main__.Int'>,
                            'default': <class 'inspect._empty'>,
                            'name': 'bb'
                        }
                    ]
                },
                {
                    'doc': None,
                    'func': <staticmethod(<function Test.func1 at 0x000001FEF5E6CD60>)>,
                    'params': []
                }
            ],
            'name': 'Test'
        }
    ] """

    def __init__(self) -> None:
        self.data = list[Mapping[str, str | list[Mapping[str, str | type | list[Mapping[str, InputWidget | str]]]]]]()

    def generate(self, parent):
        class_frame = QTabWidget(parent)
        for c in self.data:
            subtab = QTabWidget(class_frame)
            for cmd in c['cmds']:
                subtab.addTab(FuncFrame(subtab, cmd), f"{cmd['name']}")
            class_frame.addTab(subtab, f"{c['name']}")
        return class_frame

    def register(self, cls: type[CmdBase]):
        """ register """
        cmds = []
        for name, member in vars(cls).items():
            if isinstance(member, staticmethod):
                params = [
                    {
                        'name': p.name,
                        'annotation': p.annotation,
                        'default': p.default
                    } for k, p in inspect.signature(member).parameters.items()
                ]

                # print(inspect.getmodule(p['annotation']) is ui_cls for p in params)
                check_failed = [not p['annotation'] in typing.get_args(InputWidget) for p in params]
                if any(check_failed):
                    warning = f'参数类型不属于指定类型。{member} 参数类型的检查结果为 {[not b for b in check_failed]}'
                    warnings.warn(warning, RuntimeWarning)
                    continue
                
                if not member.__doc__ is None:
                    doc = '\n'.join(line.lstrip() for line in member.__doc__.split('\n')) 
                else:
                    doc = 'Here is no more information available.'
                cmds.append({
                    'doc': doc,
                    'params': params,
                    'func': member,
                    'name': name
                })

        self.data.append({
            'name': cls.__name__,
            'cmds': cmds
        })

    def mainloop(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        app = QApplication(sys.argv)
        win = self.generate(None)
        win.setWindowTitle('命令组界面 | Command Group - by ZYF')
        win.resize(700, 400)
        win.show()
        app.exec()
        # sys.exit()

    @classmethod
    def exhibit(cls):
        c = Group()

        class 测试(CmdBase):
            @staticmethod
            def 打印(参数1: UIntEditer = 0, 参数2: IntEditer = 0, 参数3: TextEditer = '', 参数4: FloatEditer = 0., /):
                """ 
                ### 尼采经典语录
                1. 自从厌倦于追寻，我已学会一觅即中;自从一股逆风袭来，我已能抗御八面来风，驾舟而行。
                1. 当你凝视深渊时，深渊也在凝视着你。
                1. 对真理而言，信服比流言更危险
                1. 白昼的光，如何能够了解夜晚黑暗的深度呢?
                1. 真正的男子渴求着不同的两件事：危险和游戏。
                1. 盲目地一味勤奋的确能创造财富和荣耀，不过，许多高尚优雅的器官也同时被这唯其能创造财富和荣耀的美德给剥夺了。
                1. 不能服从自己者便得受令于他人!
                1. 你今天是一个孤独的怪人，你离群索居，总有一天你会成为一个民族!
                1. 那些没有消灭你的东西，会使你变得更强壮
                """
                print(参数1, 参数2, 参数3, 参数4)

            @staticmethod
            def 选择(路径: PathEditer = ''):
                """
                ### 尼采经典语录
                1. 要填饱肚子，是人不能那么容易的把自己看作上帝的原因。
                1. 没有哪个胜利者信仰机遇。
                1. 鄙薄自己的人，却因此而作为鄙薄者，尊重自己。
                1. 何为生?生就是不断地把濒临死亡的威胁从自己身边抛开。
                1. 许多真理都是以笑话的形式讲出来。
                1. 要么庸俗，要么孤独。
                1. 完全不谈自己是一种甚为高贵的虚伪。
                1. 人类唯有生长在爱中，才得以创造出新的事物。
                1. 婚姻不幸福，不是因为缺乏爱，而是因为缺乏友谊。
                1. 对自己的害怕成了哲学的灵魂。
                """
                print(路径)
        c.register(测试)
        c.register(测试)
        c.mainloop()


if __name__ == '__main__':
    Group.exhibit()
