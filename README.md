## Copyright Notice
*   This code is provided solely for reviewers to reproduce the results of the paper. It shall not be used for any other purpose without the authorization of the authors.
*   This code is protected by the software copyright "Software for Predicting Isobaric Binary Vapor-Liquid Equilibrium Phase Diagrams Based on Group Contribution" (Registration No. 2024SR1229175) under the Copyright Law of the People's Republic of China. Without the authorization of the authors, no one shall copy, distribute, display, or use any part of this code. Any use of this code must comply with relevant laws and regulations, and it shall not be used for any illegal or unauthorized purpose. The authors hold no liability for any direct, indirect, incidental, special, or consequential damages arising from the use of this code.
*   Some data is derived from commercial software. We have purchased the usage rights for this software. However, public disclosure is prohibited without permission from the software provider.
*   As parts of this README were translated using a large language model, we have retained the original Chinese version, `README_CN.md`, in the folder for reference.

## Directory Tree
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
|      test_Jaubert.xlsx
│      test_newbinary.xlsx
│      test_newbinary_pred.xlsx
│      test_T_Extrapolation.xlsx
│      test_T_Extrapolation_pred.xlsx
```

## File Descriptions
*   `calc.ipynb`: A Jupyter Notebook file used to calculate the results of the test set.
*   `calc.py`: Contains the functions and classes for calculating vapor-liquid equilibrium phase diagrams.
*   `model.py`: Defines the machine learning model used for predictions.
*   `README.md`: The instruction file for the project.
*   `model`: Stores the trained machine learning model files.
*   `data`: Contains Excel files with Antoine equation parameters and molecular vector data.
    * `antoine.xlsx`: An Excel file containing Antoine equation parameters.
*   `CompareModels`: Contains Excel files, related Aspen files, and calculation files for comparing the prediction results of different models.
    * `MLPROP`: Contains the data calculated using the [MLPROP](https://ml-prop.mv.rptu.de/) website. 
    * `Aspen File`: Contains the files for calculating vapor-liquid equilibria after importing the relevant NRTL parameters into the Aspen Plus software.
    * The remaining files are the prediction results of the test set. Files ending with `_T` are the results of the temperature extrapolation test set, while the others are the prediction results of new binary pairs.
    * Files starting with `test_MLPROP_WEB` are the results calculated directly from the MLPROP website. We used a spline interpolation method to obtain the data in these files.
*  `test`: Contains the Excel files of the test set and the prediction results.
    * `test_Jaubert.xlsx`: The complete test results of MFSNN on the database proposed by Jaubert et al.
    * `test_newbinary.xlsx`: The input data of the new binary pair test set.
    * `test_newbinary_pred.xlsx`: The prediction results of the new binary pair test set.
    * `test_T_Extrapolation.xlsx`: The input data of the temperature extrapolation test set.
    * `test_T_Extrapolation_pred.xlsx`: The prediction results of the temperature extrapolation test set.

## Required Libraries
```python
pip install -r requirements.txt
```

## Key Notes for Usage
1. Due to copyright issues, the `antoine.xlsx` file is not publicly available. Users must first replace the zero parameters in the file with valid vapor pressure equation parameters. The calculation formula is:
$$
    \ln p^{\text{sat}}/\text{Pa} = A + \frac{B}{T + C} + D \times T + E \times \ln T + F \times T^G
$$
2. The `calc.ipynb` file contains the code for calculating test set results. Users can run this file directly to obtain the predicted results for the test set.
3. The `calc.py` file contains functions and classes for calculating vapor-liquid equilibrium phase diagrams. Users can call these functions and classes as needed to perform calculations.