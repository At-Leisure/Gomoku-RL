All the documentations is in `docs/` folder.

|   窗  |   口   |   图   |   标   |
| :----: | :-----: |:-----: |:-----: |
| ![](../_internal/gui/ico/document.png) | ![](../_internal/gui/ico/v.png) |![](../_internal/gui/ico/code.png) |![](../_internal/gui/ico/list2.png) |

## 编程版本

|   工具  |   版本   |
| :----: | :-----: |
| python | 3.10.14 |


## 决策流程

![](./网络决策.png)

## 图形化

![](./展示GUI.png)

## 命令行

```yaml
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  五子棋命令行

Options:
  --help  Show this message and exit.

Commands:
  agent-self-play   AI自博弈
  draw-network-io   绘制AI模型的输入输出分析示意图
  graphic-control   PyQt6界面-简化命令行
  play-with-agent   与AI博弈
  play-with-human   与人博弈
  show-frame-img    测试GUI的渲染效果
  train-from-model  使用预训练模型对AI进行训练
  train-with-empty  从零开始对AI进行训练
```


[MCTS参考链接](https://www.cnblogs.com/TABball/p/12727130.html)