## 版权声明
*   该部分代码仅提供审稿人复现论文结果之用，未经作者授权，不得用于任何其他目的。
*   本部分代码享有软件著作权“基于基团贡献的常压二元气液相平衡相图预测软件”（2024SR1229175），受中华人民共和国著作权法的保护。未经作者授权，任何人不得复制、分发、展示或使用本代码的任何部分。任何使用本代码的行为必须遵守相关法律法规，并且不得用于任何非法或未经授权的目的。对于因使用本代码而产生的任何直接、间接、偶然、特殊或后果性损害，作者不承担任何责任。
*   部分数据来源于商业软件，我方已经购买该商业软件的使用权，未经商业软件提供商允许禁止公开。

## 目录树
```
.
│  calc.ipynb
│  calc.py
│  model.py
│  README.md
│
├─CompareModels
│  │  test_MLPROP.xlsx
│  │  test_MLPROP_T.xlsx
│  │  test_MLPROP_WEB.xlsx
│  │  test_MLPROP_WEB_T.xlsx
│  │  test_SPT_NRTL.xlsx
│  │  test_SPT_NRTL_T.xlsx
│  │  test_UNIFAC.xlsx
│  │  test_UNIFAC_T.xlsx
│  │
│  ├─Aspen File
│  │  MLPROPNRTL.apwz
│  │  SPFNRTL.apwz
│  │  UNIFAC.apwz
│  │
│  └─MLPROP
│     1,1,1-TRIFLUOROETHANE_ISOBUTYLENE_T.csv
│     1-PROPANOL_CHLOROFORM_P.csv
│     BENZENE_PERFLUOROBENZENE_P.csv
│     interpt.ipynb
│     METHANOL_2-BUTANOL_P_101300.csv
│     METHANOL_2-BUTANOL_P_101330.csv
│     N-HEXANE_1,4-DIOXANE_T.csv
│     N-HEXANE_1-CHLOROPENTANE_T.csv
│     N-HEXANE_1-DECENE_P.csv
│     N-HEXANE_METHYL-ACETATE_P.csv
│     N-PROPIONALDEHYDE_ETHANOL_P.csv
│     PYRIDINE_ISOBUTANOL_P.csv
│
├─data
│      antoine.xlsx
│      mol_vector.xlsx
│
├─model
│      model_1060306957.pt
│      model_1415522601.pt
│      model_246358563.pt
│      model_26317076.pt
│      model_26318864.pt
│      model_270358073.pt
│      model_539035672.pt
│      model_701857512.pt
│      model_870122900.pt
│
├─test
│      test_newbinary.xlsx
│      test_newbinary_pred.xlsx
│      test_T_Extrapolation.xlsx
│      test_T_Extrapolation_pred.xlsx
```

## 文件说明
*   `calc.ipynb`：计算测试集结果的Jupyter Notebook文件。
*   `calc.py`：包含计算气液平衡相图的函数和类。
*   `model.py`：定义了用于预测的机器学习模型。
*   `README.md`：项目的说明文件。
*   `model`：存储了训练好的机器学习模型文件。
*   `data`：包含了安托万方程参数和分子向量数据的Excel文件。
    * `antoine.xlsx`：包含了安托万方程参数的Excel文件。
*   `CompareModels`：包含了比较不同模型预测结果的Excel文件和相关的Aspen文件、计算文件。
    * `MLPROP`：包含使用[MLPROP](https://ml-prop.mv.rptu.de/)网页计算得到的数据。 
    * `Aspen File`：包含相关NRTL参数导入Aspen Plus软件后计算气液相平衡的文件。
    * 其余文件均为测试集的预测结果，以`_T`结尾的文件为温度外推测试集的结果，其他均为新二元对的预测结果。
    * `test_MLPROP_WEB`开头的文件为直接从MLPROP网页上计算得到的结果，我们使用样条插值的方法计算得到了该文件数据。
*  `test`：包含了测试集的Excel文件和预测结果的Excel文件。
    * `test_newbinary.xlsx`：新二元对测试集的输入数据。
    * `test_newbinary_pred.xlsx`：新二元对测试集的预测结果。
    * `test_T_Extrapolation.xlsx`：温度外推测试集的输入数据。
    * `test_T_Extrapolation_pred.xlsx`：温度外推测试集的预测结果。

## 需要安装的库
```python
pip install -r requirements.txt
```

## 使用主要注意事项
1. 由于版权问题，`antoine.xlsx`文件无法公开，读者需要首先将其中的0参数替换为可用的蒸气压方程参数，其计算规则为：
$$
    \ln p^{\text{sat}}/\text{Pa} = A + \frac{B}{T + C} + D \times T + E \times \ln T + F \times T^G
$$
2. `calc.ipynb`文件中包含了计算测试集结果的代码，读者可以直接运行该文件来计算测试集的预测结果。
3. `calc.py`文件中包含了计算气液平衡相图的函数和类，读者可以根据需要调用这些函数和类来进行计算。