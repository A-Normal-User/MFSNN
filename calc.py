import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from model import Net
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.stats import shapiro
from scipy.spatial import ConvexHull
import scipy.integrate
import scipy.interpolate
from warnings import catch_warnings
num_group = 53
def split_smiles(smiles):
    smarts_fragments = [
        "[CH4]",                                          # 1 CH4
        "[CX4H3]",                                        # 2 CH3-
        "[CX4H2]",                                        # 3 -CH2-
        "[CX4H]",                                         # 4 >CH-
        "[CX4H0]",                                        # 5 >C<
        "[CX3H2]=[CX3H2]",                                # 6 CH2=CH2
        "[CX3H1]=[CX3H2]",                                # 7 -CH=CH2
        "[CX3H1]=[CX3H1]",                                # 8 -CH=CH-
        "[CX3H0]=[CX3H2]",                                # 9 >C=CH2
        "[CX3H0]=[CX3H1]",                                # 10 >C=CH-
        "[CX3H0]=[CX3H0]",                                # 11 >C=C<
        "[cH]",                                           # 12 c-H (芳环碳氢)
        "[CX2H]#[CX2H]",                                  # 13 HC≡CH
        "[CX2H]#[CX2H0]",                                 # 14 HC≡C-
        "[CX2H0]#[CX2H0]",                                # 15 -C≡C-
        "[cH0]",                                          # 16 c (芳环碳)
        "[CX3](=O)[OH]",                                  # 17 -COOH
        "[CX3H1;!$(C-O)](=O)",                            # 18 -CHO
        "[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]",        # 19 -OH (alcohol)
        "C(=O)O[H]",                                      # 20 HCOOH
        "[CX3H](=O)[OX2]",                                # 21 HCOO-
        "[CX3H0;!$([C]-[O]-[C](=O))](=O)[OX2H0]",         # 22 -COO-
        "[BrX1H0]",                                       # 23 -Br
        "[IX1H0]",                                        # 24 -I
        "[H]C#N",                                         # 25 HCN
        "[H][Cl]",                                        # 26 HCl
        "[H][F]",                                         # 27 HF
        "[H][Br]",                                        # 28 HBr
        "[H][I]",                                         # 29 HI
        "[NH3]",                                          # 30 NH3
        "[OH2]",                                          # 31 H2O
        "[SH2]",                                          # 32 H2S
        "[OX2H0;!$([OX2H0]-[C]=[O])]",                    # 33 -O-
        "[$([CX3H0](=[OX1]));!$([CX3](=[OX1])~[OX2])]=O", # 34 >C=O
        "[NX3H]",                                         # 35 -NH-
        "[ClX1H0]",                                       # 36 -Cl
        "[FX1H0]",                                        # 37 -F
        "[NX3H2]",                                        # 38 -NH2
        "[NX3H0;!$([N](~O)~O)]",                          # 39 -N<
        "[N+](=O)[O-]",                                   # 40 -NO2
        "[CX2H0]#N",                                      # 41 -CN
        "[SX2H]",                                         # 42 -SH
        "[n]",                                            # 43 n (芳环氮)
        "[o]",                                            # 44 o (芳环氧)
        "[SX3](=O)",                                      # 45 >S=O
        "[SiX4]",                                         # 46 >Si<
        "[s]",                                            # 47 s
        "[SX4H0](=O)(=O)",                                # 48 (O=)S<(=O)
        "[OX2][OX2]",                                     # 49 -O-O-
        "[SX2H0]",                                        # 50 -S-
        "[N]=[C]=[O]",                                    # 51 -N=C=O
        "O=C-O-C=O",                                      # 52 -C=O-C=O
        "[O;H1;$(O-!@c)]",                                # 53 -OH (phenol)
    ]
    # 根据 SMILES 字符串创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 补充分子对象的氢原子
    mol = Chem.AddHs(mol)
    # 计算 SMIELS式中smarts_fragments每个片段的数量
    counts = []
    for smarts in smarts_fragments:
        pattern = Chem.MolFromSmarts(smarts)
        counts.append(len(mol.GetSubstructMatches(pattern)))
    return counts

def nrtl_CEM(tau12, tau21, alpha):

    N = 10001                           # 网格点数
    x1 = np.linspace(1e-6, 1.0 - 1e-6, N)   # 组分1摩尔分数，避开端点
    x2 = 1.0 - x1
    g12 = np.exp(-alpha * tau12)
    g21 = np.exp(-alpha * tau21)
    gE_RT = x1 * x2 * (tau21 * g21 / (x1 + x2 * g21) +
                       tau12 * g12 / (x2 + x1 * g12))
    g_ideal = x1 * np.log(x1) + x2 * np.log(x2)
    g_mix = g_ideal + gE_RT          # 总混合 Gibbs 自由能（g/RT）
    pts = np.column_stack((x1, g_mix))
    hull = ConvexHull(pts)

    verts = hull.vertices
    # 将凸包顶点按 x 升序排列
    verts_sorted = verts[np.argsort(x1[verts])]

    # 在下边界上运行 Andrew 单调链，得到保留的顶点索引（原始网格索引）
    def lower_hull(xs, gs, idxs):
        lower = []
        for i in range(len(xs)):
            while len(lower) >= 2:
                o, a = lower[-2], lower[-1]
                cross = ((xs[a] - xs[o]) * (gs[i] - gs[o]) -
                         (gs[a] - gs[o]) * (xs[i] - xs[o]))
                if cross <= 0:            # 右转或共线则弹出
                    lower.pop()
                else:
                    break
            lower.append(i)
        return [idxs[i] for i in lower]

    lower = lower_hull(x1[verts_sorted], g_mix[verts_sorted], verts_sorted)

    best_gap = 0
    tie = (None, None)
    for i in range(len(lower) - 1):
        ia, ib = lower[i], lower[i + 1]
        gap = x1[ib] - x1[ia]
        # 原网格上不相邻（至少跳过一个点）且跨度更大
        if (ib - ia) > 1 and gap > best_gap:
            xa, xb = x1[ia], x1[ib]
            ga, gb = g_mix[ia], g_mix[ib]
            mid = np.arange(ia + 1, ib)             # 被跳过的中间点索引
            g_line = ga + (gb - ga) * (x1[mid] - xa) / (xb - xa)
            # 验证整段连线都在原曲线下方（或贴合）
            if np.all(g_line <= g_mix[mid] + 1e-12):
                best_gap = gap
                tie = (xa, xb)

    # 返回结果：是否存在液液分相，以及两平衡相的 x1
    if tie[0] is None:
        return False, np.nan, np.nan
    else:
        return True, tie[0], tie[1]

def predict_ANN(model:Net, parameter:torch.Tensor) -> np.ndarray:
    """
    Predict the result using the ANN model.
    model: list[Net], list of ANN models.
    parameter: np.ndarray, input parameters for the model.
    Returns:
        - np.ndarray, predicted result.
    """
    # Convert the input parameter to a PyTorch tensor
    model.eval()
    # Get the prediction
    with torch.no_grad():
        predictions = model(parameter).numpy()
    return predictions

def predict_tau_alpha(model:Net, parameter:torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict the activity coefficient using the ANN model.
    model: list[Net], list of ANN models.
    parameter: np.ndarray, input parameters for the model.
    x1: np.ndarray, input parameters for the model.
    Returns:
        - tuple[np.ndarray, np.ndarray], predicted activity coefficient and its standard deviation.
    """
    pred = predict_ANN(model, parameter)
    # Calculate the activity coefficient using the predicted values
    tau12, tau21, alpha = pred[:, 0], pred[:, 1], pred[:, 2]
    return (tau12, tau21, alpha)

def predict_gamma(model:Net, parameter:torch.Tensor, x1:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the activity coefficient using the ANN model.
    model: list[Net], list of ANN models.
    parameter: np.ndarray, input parameters for the model.
    x1: np.ndarray, input parameters for the model.
    Returns:
        - tuple[np.ndarray, np.ndarray], predicted activity coefficient and its standard deviation.
    """
    pred = predict_ANN(model, parameter)
    # Calculate the activity coefficient using the predicted values
    tau12, tau21, alpha = pred[:, 0], pred[:, 1], pred[:, 2]
    g12 = np.exp(-alpha * tau12)
    g21 = np.exp(-alpha * tau21)
    x2 = 1 - x1
    gamma1 = np.exp(x2**2 * (tau21 * g21**2 / (x1 + x2 * g21)**2 + tau12 * g12 / (x2 + x1 * g12)**2))
    gamma2 = np.exp(x1**2 * (tau12 * g12**2 / (x2 + x1 * g12)**2 + tau21 * g21 / (x1 + x2 * g21)**2))
    return (gamma1, gamma2)

def predict_LLE_S(alpha:np.ndarray, tau12:np.ndarray, tau21:np.ndarray, x1:np.ndarray) -> np.ndarray:
    """
    Predict the LLE Function using the ANN model. (Spinodal line)
    alpha: np.ndarray, predicted alpha values.
    tau12: np.ndarray, predicted tau12 values.
    tau21: np.ndarray, predicted tau21 values.
    x1: np.ndarray, input parameters for the model.
    Returns:
        - np.ndarray, predicted LLE Function.
    """
    g12 = np.exp(-alpha * tau12)
    g21 = np.exp(-alpha * tau21)
    x2 = 1 - x1
    F = 1
    F -= 2 * x1 * x2 * (tau21 * (g21 ** 2) / ((x1 + x2 * g21) ** 3) + tau12 * (g12 ** 2) / ((x2 + x1 * g12) ** 3))
    return F

def predict_LLE_B(alpha:np.ndarray, tau12:np.ndarray, tau21:np.ndarray, x1:np.ndarray) -> np.ndarray:
    """
    Predict the LLE Function using the ANN model. (Binodal line)
    alpha: np.ndarray, predicted alpha values.
    tau12: np.ndarray, predicted tau12 values.
    tau21: np.ndarray, predicted tau21 values.
    x1: np.ndarray, input parameters for the model.
    Returns:
        - np.ndarray, predicted LLE Function.
    """
    g12 = np.exp(-alpha * tau12)
    g21 = np.exp(-alpha * tau21)
    x2 = 1 - x1
    gamma1 = np.exp(x2**2 * (tau21 * g21**2 / (x1 + x2 * g21)**2 + tau12 * g12 / (x2 + x1 * g12)**2))
    gamma2 = np.exp(x1**2 * (tau12 * g12**2 / (x2 + x1 * g12)**2 + tau21 * g21 / (x1 + x2 * g21)**2))
    return np.concatenate([gamma1 * x1, gamma2 * (1 - x1)])

def predict_P_gamma(gamma1:np.ndarray, gamma2:np.ndarray, x1:np.ndarray, t:np.ndarray, vp1, vp2) -> np.ndarray:
    """
    Predict the vapor pressure using the ANN model.
    gamma1: np.ndarray, activity coefficient of the first component.
    gamma2: np.ndarray, activity coefficient of the second component.
    x1: np.ndarray, input parameters for the model.
    t: np.ndarray, temperature in Kelvin.
    vp1: function, vapor pressure function of the first component.
    vp2: function, vapor pressure function of the second component.
    Returns:
        - tuple[np.ndarray, np.ndarray], predicted vapor pressure and its standard deviation.
    """
    return (np.exp(vp1(t)) * x1 * gamma1 + np.exp(vp2(t)) * (1 - x1) * gamma2)

def predict_P(model:Net, parameter:torch.Tensor, x1:np.ndarray, t:np.ndarray, vp1, vp2) -> np.ndarray:
    """
    Predict the vapor pressure using the ANN model.
    gamma1: np.ndarray, activity coefficient of the first component.
    gamma2: np.ndarray, activity coefficient of the second component.
    x1: np.ndarray, input parameters for the model.
    t: np.ndarray, temperature in Kelvin.
    vp1: function, vapor pressure function of the first component.
    vp2: function, vapor pressure function of the second component.
    Returns:
        - tuple[np.ndarray, np.ndarray], predicted vapor pressure and its standard deviation.
    """
    parameter[:, num_group*2:] = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    pred = predict_ANN(model, parameter)
    # Calculate the activity coefficient using the predicted values
    tau12, tau21, alpha = pred[:, 0], pred[:, 1], pred[:, 2]
    g12 = np.exp(-alpha * tau12)
    g21 = np.exp(-alpha * tau21)
    x2 = 1 - x1
    gamma1 = np.exp(x2**2 * (tau21 * g21**2 / (x1 + x2 * g21)**2 + tau12 * g12 / (x2 + x1 * g12)**2))
    gamma2 = np.exp(x1**2 * (tau12 * g12**2 / (x2 + x1 * g12)**2 + tau21 * g21 / (x1 + x2 * g21)**2))
    return (np.exp(vp1(t)) * x1 * gamma1 + np.exp(vp2(t)) * (1 - x1) * gamma2)

def log_extended_antoine(t, params):
    """
    Extended Antoine equation for vapor pressure calculation.
    t: float, temperature in Kelvin.
    params: list, parameters for the extended Antoine equation.
        - [a, b, c, d, e, f, g]
    Returns:
        - float, logarithmic vapor pressure in log(Pa).
    """
    # Check if the length of params is 7
    if len(params) != 7:
        raise ValueError("params must be a list of length 7.")
    # Unpack the parameters
    a, b, c, d, e, f, g = params
    return a + b / (t + c) + d * t + e * np.log(t) + f * (t ** g)

class BGMLVLE():
    def __init__(self, is_smiles:bool = False, vapor_pressure = None) -> None:
        """
        Initialize the BGMLVLE class.
        is_smiles: bool, if True, the molecules input must be a SMILES string.
        vapor_pressure: function, Function for calculating the logarithmic vapor pressure of a component.
            - This function should take a molecule and a temperature as input and return the logarithmic vapor pressure.
        if is_smiles is True, vapor_pressure_function must be provided.
        """
        self.is_smiles = is_smiles
        self.vapor_pressure = vapor_pressure
        if is_smiles == True and vapor_pressure == None:
            raise ValueError("If is_smiles is True, vapor_pressure_function must be provided.")
        
        self.extended_antoine = None
        if vapor_pressure == None:
            try:
                self.extended_antoine = pd.read_excel('./data/antoine.xlsx')
            except FileNotFoundError:
                print("antoine.xlsx file not found. Please provide the vapor pressure function.")
        
        self.model_file = [
            'model_26317076.pt',
            'model_26318864.pt',
            'model_246358563.pt',
            'model_270358073.pt',
            'model_539035672.pt',
            'model_701857512.pt',
            'model_870122900.pt',
            'model_1060306957.pt',
            'model_1415522601.pt',
        ]
        self.model = []
        for i in range(len(self.model_file)):
            self.model.append(Net())
            self.model[i].load_state_dict(torch.load(f'./model/{self.model_file[i]}', weights_only=True))
            self.model[i] = torch.jit.script(self.model[i])
        
        self.mol_frame = None
        if is_smiles == False:
            self.mol_frame = pd.read_excel('./data/mol_vector.xlsx')

    def Isobaric(self, molecule1:str, molecule2:str, P:float, x:list[float], normality_test:bool = False, thermodynamic_consistency_test:bool = False) -> tuple[list[list[float]], list[list[float]]]:
        """
        Calculate the bubble point and dew point of a binary mixture at a given pressure.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        P: float, pressure in atm.
        x: list[float], mole fraction of the first component in the liquid phase.
        normality_test: bool, if True, perform normality test.
        thermodynamic_consistency_test: bool, if True, perform thermodynamic consistency test.
        Returns:
            - y: list[list[float]], mole fraction of the first component in the vapor phase.
                - y[0]: Gas phase mole fraction
                - y[1]: The lower bound of the 95% confidence interval of the gas phase mole fraction
                - y[2]: The upper bound of the 95% confidence interval of the gas phase mole fraction
            - T_bubble: list[list[float]], bubble point temperature in K.
                - T_bubble[0]: Bubble point temperature
                - T_bubble[1]: The lower bound of the 95% confidence interval of the bubble point temperature
                - T_bubble[2]: The upper bound of the 95% confidence interval of the bubble point temperature
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)
        
        vp1, vp2, params1, params2 = None, None, None, None
        if self.vapor_pressure == None:
            # Get the parameters from the extended Antoine equation
            index1 = self.extended_antoine[self.extended_antoine['Description'] == molecule1].index[0]
            index2 = self.extended_antoine[self.extended_antoine['Description'] == molecule2].index[0]
            params1 = self.extended_antoine.iloc[index1, 1:].values.tolist()
            params2 = self.extended_antoine.iloc[index2, 1:].values.tolist()
            vp1 = lambda t: log_extended_antoine(t, params1)
            vp2 = lambda t: log_extended_antoine(t, params2)
        else:
            vp1 = lambda t: self.vapor_pressure(molecule1, t)
            vp2 = lambda t: self.vapor_pressure(molecule2, t)

        # Calculate the end point temperature based on P
        with catch_warnings(record=True) as w:
            T1_end = fsolve(lambda t: vp2(t) - np.log(P), 273.15)[0]
            if len(w) > 0:
                print(f"Warning: {w[0].message}, the vapor pressure may be not valid for the given pressure.")
        with catch_warnings(record=True) as w:
            T2_end = fsolve(lambda t: vp1(t) - np.log(P), 273.15)[0]
            if len(w) > 0:
                print(f"Warning: {w[0].message}, the vapor pressure may be not valid for the given pressure.")

        # Calculate the bubble point temperature and dew point temperature
        T_bubble = []
        y = []
        gammas1, gammas2 = [], []
        param = torch.zeros((x.shape[0], num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        is_error = False
        for i in range(len(self.model)):
            # starting temperature for the calculation
            # Calculate the bubble point temperature and dew point temperature
            t0 = np.linspace(T1_end, T2_end, x.shape[0], dtype=np.float32).reshape(-1)
            with catch_warnings(record=True) as w:
                t0 = fsolve(lambda t: np.log(predict_P(self.model[i], param, x, t, vp1, vp2)) - np.log(P), t0)
                param[:, num_group*2:] = torch.tensor(t0, dtype=torch.float32).unsqueeze(1)
                gamma1, gamma2 = predict_gamma(self.model[i], param, x)
                if len(w) > 0:
                    is_error = True
                T_bubble.append(t0)
                y.append((np.exp(vp1(t0)) * x * gamma1) / P)
                gammas1.append(gamma1)
                gammas2.append(gamma2)
        if is_error == True:
            print("Warning: The calculation may not be valid for the given pressure.")
        y = np.array(y)
        T_bubble = np.array(T_bubble)
        if normality_test:
            # shapiro正态性检验
            shapiro_pass = 0
            for i in range(len(x)):
                if shapiro(T_bubble[:, i])[1] > 0.05:
                    shapiro_pass += 1
            print(f'shapiro test T_bubble: {shapiro_pass} / {len(x)}')
            shapiro_pass = 0
            for i in range(len(x)):
                if shapiro(y[:, i])[1] > 0.05:
                    shapiro_pass += 1
            print(f'shapiro test y: {shapiro_pass} / {len(x)}')
        # Calculate the 95% confidence interval
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_lower = y_mean - 2.306 * y_std / 3
        y_upper = y_mean + 2.306 * y_std / 3
        T_bubble_mean = np.mean(T_bubble, axis=0)
        T_bubble_std = np.std(T_bubble, axis=0)
        T_bubble_lower = T_bubble_mean - 2.306 * T_bubble_std / 3
        T_bubble_upper = T_bubble_mean + 2.306 * T_bubble_std / 3
        # Thermodynamic consistency test
        if thermodynamic_consistency_test:
            gammas1 = np.array(gammas1).mean(axis=0)
            gammas2 = np.array(gammas2).mean(axis=0)
            # 计算热力学一致性
            x = np.array(x, dtype=np.float32)
            newCalc = scipy.interpolate.interp1d(x, np.log(gamma1 / gamma2), kind = "cubic", fill_value = 'extrapolate')(x)
            delta = np.abs(scipy.integrate.simpson(newCalc, x)) / scipy.integrate.simpson(np.abs(newCalc), x)
            J = 150 * np.abs(T_bubble_mean.max() - T_bubble_mean.min()) / T_bubble_mean.min()
            D = 100 * delta
            R = np.abs(D - J)
            A_star = 100 * scipy.integrate.simpson(newCalc, x)
            print(f"Thermodynamic consistency test: D = {D}, J = {J}, R = {R}, A* = {A_star}, if R < 10 or |A*| < 3 means Herington Test pass.")
        # Return the results
        return [y_mean, y_lower, y_upper], [T_bubble_mean, T_bubble_lower, T_bubble_upper]
    
    def Isothermal(self, molecule1:str, molecule2:str, T:float, x:list[float], normality_test:bool = False, thermodynamic_consistency_test:bool = False) -> tuple[list[list[float]], list[list[float]]]:
        """
        Calculate the bubble point and dew point of a binary mixture at a given temperature.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        T: float, temperature in K.
        x: list[float], mole fraction of the first component in the liquid phase.
        normality_test: bool, if True, perform normality test.
        thermodynamic_consistency_test: bool, if True, perform thermodynamic consistency test.
        Returns:
            - y: list[list[float]], mole fraction of the first component in the vapor phase.
                - y[0]: Gas phase mole fraction
                - y[1]: The lower bound of the 95% confidence interval of the gas phase mole fraction
                - y[2]: The upper bound of the 95% confidence interval of the gas phase mole fraction
            - P: list[list[float]], bubble point pressure in Pa.
                - P[0]: Bubble point pressure
                - P[1]: The lower bound of the 95% confidence interval of the bubble point pressure
                - P[2]: The upper bound of the 95% confidence interval of the bubble point pressure
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)
        
        vp1, vp2, params1, params2 = None, None, None, None
        if self.vapor_pressure == None:
            # Get the parameters from the extended Antoine equation
            index1 = self.extended_antoine[self.extended_antoine['Description'] == molecule1].index[0]
            index2 = self.extended_antoine[self.extended_antoine['Description'] == molecule2].index[0]
            params1 = self.extended_antoine.iloc[index1, 1:].values.tolist()
            params2 = self.extended_antoine.iloc[index2, 1:].values.tolist()
            vp1 = lambda t: log_extended_antoine(t, params1)
            vp2 = lambda t: log_extended_antoine(t, params2)
        else:
            vp1 = lambda t: self.vapor_pressure(molecule1, t)
            vp2 = lambda t: self.vapor_pressure(molecule2, t)

        param = torch.zeros((len(x), num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32)
        y = []
        P_bubble = []
        gammas1, gammas2 = [], []
        for i in range(len(self.model)):
            # starting temperature for the calculation
            gamma1, gamma2 = predict_gamma(self.model[i], param, x)
            # Calculate the bubble point temperature and dew point temperature
            P = predict_P_gamma(gamma1, gamma2, x, T, vp1, vp2)
            P_bubble.append(P)
            y.append((np.exp(vp1(T)) * x * gamma1) / P)
            gammas1.append(gamma1)
            gammas2.append(gamma2)
        y = np.array(y)
        P_bubble = np.array(P_bubble)
        if normality_test:
            # shapiro正态性检验
            shapiro_pass = 0
            for i in range(len(x)):
                if shapiro(P_bubble[:, i])[1] > 0.05:
                    shapiro_pass += 1
            print(f'shapiro test P_bubble: {shapiro_pass} / {len(x)}')
            shapiro_pass = 0
            for i in range(len(x)):
                if shapiro(y[:, i])[1] > 0.05:
                    shapiro_pass += 1
            print(f'shapiro test y: {shapiro_pass} / {len(x)}')
        # Calculate the 95% confidence interval
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_lower = y_mean - 2.306 * y_std / 3
        y_upper = y_mean + 2.306 * y_std / 3
        P_bubble_mean = np.mean(P_bubble, axis=0)
        P_bubble_std = np.std(P_bubble, axis=0)
        P_bubble_lower = P_bubble_mean - 2.306 * P_bubble_std / 3
        P_bubble_upper = P_bubble_mean + 2.306 * P_bubble_std / 3
        # Thermodynamic consistency test
        if thermodynamic_consistency_test:
            gammas1 = np.array(gammas1).mean(axis=0)
            gammas2 = np.array(gammas2).mean(axis=0)
            # 计算热力学一致性
            x = np.array(x, dtype=np.float32)
            newCalc = scipy.interpolate.interp1d(x, np.log(gamma1 / gamma2), kind = "cubic", fill_value = 'extrapolate')(x)
            delta = np.abs(scipy.integrate.simpson(newCalc, x)) / scipy.integrate.simpson(np.abs(newCalc), x)
            J = 0
            D = 100 * delta
            R = np.abs(D - J)
            A_star = 100 * scipy.integrate.simpson(newCalc, x)
            print(f"Thermodynamic consistency test: D = {D}, J = {J}, R = {R}, A* = {A_star}, if R < 10 or |A*| < 3 means Herington Test pass.")
        # Return the results
        return [y_mean, y_lower, y_upper], [P_bubble_mean, P_bubble_lower, P_bubble_upper]
    
    def LLE(self, molecule1:str, molecule2:str, T:float) -> tuple[list[list[float]], list[list[float]]]:
        """
        Calculate the liquid-liquid equilibrium of a binary mixture at a given temperature.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        T: float, temperature in K.
        Returns:
            - xL1: list[list[float]], mole fraction of the first component in the liquid phase.
                - xL1[0]: Mole fraction of the first component in the liquid phase
                - xL1[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
                - xL1[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
            - xL2: list[list[float]], mole fraction of the first component in the vapor phase.
                - xL2[0]: Mole fraction of the first component in the vapor phase
                - xL2[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
                - xL2[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)

        param = torch.zeros((1, num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32)
        x = np.linspace(0, 1, 100, dtype=np.float32)
        xL1 = []
        xL2 = []
        for i in range(len(self.model)):
            tau12, tau21, alpha = predict_tau_alpha(self.model[i], param)
            S = predict_LLE_S(alpha, tau12, tau21, x)
            print(f'Model_{i+1}_S = {S.tolist()}')
            # 如果S恒为正数，则说明没有两相平衡，直接返回空结果
            if np.all(S > 0):
                xL1.append(np.nan)
                xL2.append(np.nan)
                continue
            # 从0开始找到S的近似变号点，从1开始找到另外一个S的近似变号点(旋节线)
            xL1_0 = 0
            xL2_0 = 1
            for j in range(len(x)-1):
                if S[j] * S[j+1] < 0:
                    xL1_0 = x[j]
                    break
            for j in range(len(x)-1, 0, -1):
                if S[j] * S[j-1] < 0:
                    xL2_0 = x[j]
                    break
            # 使用fsolve找到更精确的变号点(双节线)
            with catch_warnings(record=True) as w:
                xL = least_squares(lambda x0: predict_LLE_B(alpha, tau12, tau21, x0[0]) - predict_LLE_B(alpha, tau12, tau21, x0[1]), x0=(xL1_0, xL2_0), bounds=([0,0], [1,1]), ftol=1e-12).x
                if len(w) > 0:
                    print(f"Warning: {w[0].message}, the calculation may not be valid for the given temperature.")
            xL1.append(xL[0])
            xL2.append(xL[1])
        xL1 = np.array(xL1)
        xL2 = np.array(xL2)
        xL1_mean = np.nanmean(xL1)
        xL2_mean = np.nanmean(xL2)
        xL1_std = np.nanstd(xL1, ddof=1)
        xL2_std = np.nanstd(xL2, ddof=1)
        xL1_lower = xL1_mean - 2.306 * xL1_std / np.sqrt(len(xL1))
        xL1_upper = xL1_mean + 2.306 * xL1_std / np.sqrt(len(xL1))
        xL2_lower = xL2_mean - 2.306 * xL2_std / np.sqrt(len(xL2))
        xL2_upper = xL2_mean + 2.306 * xL2_std / np.sqrt(len(xL2))
        # Return the results
        return [xL1_mean, xL1_lower, xL1_upper], [xL2_mean, xL2_lower, xL2_upper]
    
    def LLE_fug_average(self, molecule1:str, molecule2:str, T:float) -> tuple[float, float]:
        """
        Calculate the liquid-liquid equilibrium of a binary mixture at a given temperature.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        T: float, temperature in K.
        Returns:
            - xL1: list[list[float]], mole fraction of the first component in the liquid phase.
                - xL1[0]: Mole fraction of the first component in the liquid phase
                - xL1[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
                - xL1[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
            - xL2: list[list[float]], mole fraction of the first component in the vapor phase.
                - xL2[0]: Mole fraction of the first component in the vapor phase
                - xL2[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
                - xL2[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)
        if self.vapor_pressure == None:
            # Get the parameters from the extended Antoine equation
            index1 = self.extended_antoine[self.extended_antoine['Description'] == molecule1].index[0]
            index2 = self.extended_antoine[self.extended_antoine['Description'] == molecule2].index[0]
            params1 = self.extended_antoine.iloc[index1, 1:].values.tolist()
            params2 = self.extended_antoine.iloc[index2, 1:].values.tolist()
            vp1 = lambda t: log_extended_antoine(t, params1)
            vp2 = lambda t: log_extended_antoine(t, params2)
        else:
            vp1 = lambda t: self.vapor_pressure(molecule1, t)
            vp2 = lambda t: self.vapor_pressure(molecule2, t)

        param = torch.zeros((1, num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32)
        xL1 = []
        xL2 = []
        def _LLE_B(xLL):
            xL1_1, xL2_1 = xLL
            yL1_ = []
            p1_ = []
            yL2_ = []
            p2_ = []
            vap1 = np.exp(vp1(T))
            vap2 = np.exp(vp2(T))
            for i in range(len(self.model)):
                tau12, tau21, alpha = predict_tau_alpha(self.model[i], param)
                g12 = np.exp(-alpha * tau12)
                g21 = np.exp(-alpha * tau21)
                xL1_2 = 1 - xL1_1
                gammaL1_1 = np.exp(xL1_2**2 * (tau21 * g21**2 / (xL1_1 + xL1_2 * g21)**2 + tau12 * g12 / (xL1_2 + xL1_1 * g12)**2))
                gammaL1_2 = np.exp(xL1_1**2 * (tau12 * g12**2 / (xL1_2 + xL1_1 * g12)**2 + tau21 * g21 / (xL1_1 + xL1_2 * g21)**2))
                xL2_2 = 1 - xL2_1
                gammaL2_1 = np.exp(xL2_2**2 * (tau21 * g21**2 / (xL2_1 + xL2_2 * g21)**2 + tau12 * g12 / (xL2_2 + xL2_1 * g12)**2))
                gammaL2_2 = np.exp(xL2_1**2 * (tau12 * g12**2 / (xL2_2 + xL2_1 * g12)**2 + tau21 * g21 / (xL2_1 + xL2_2 * g21)**2))
                p1 = (vap1 * xL1_1 * gammaL1_1 + vap2 * xL1_2 * gammaL1_2)
                p2 = (vap1 * xL2_1 * gammaL2_1 + vap2 * xL2_2 * gammaL2_2)
                yL1 = vap1 * xL1_1 * gammaL1_1 / (vap1 * xL1_1 * gammaL1_1 + vap2 * xL1_2 * gammaL1_2)
                yL2 = vap1 * xL2_1 * gammaL2_1 / (vap1 * xL2_1 * gammaL2_1 + vap2 * xL2_2 * gammaL2_2)
                yL1_.append(yL1)
                yL2_.append(yL2)
                p1_.append(p1)
                p2_.append(p2)
            yL1_ = np.array(yL1_).mean()
            yL2_ = np.array(yL2_).mean()
            p1_ = np.array(p1_).mean()
            p2_ = np.array(p2_).mean()
            # print(f"xL1: {xL1_1}, xL2: {xL2_1}, yL1: {yL1_}, yL2: {yL2_}")
            return [(yL1_ - yL2_) ** 2, (np.log(p1_) - np.log(p2_)) ** 2]
        xL1 = 0
        xL2 = 1
        with catch_warnings(record=True) as w:
            # constraints = {'type': 'ineq', 'fun': lambda p: p[1] - p[0] - 1e-3}  # Ensure xL2 >= xL1 + 1e-2
            # fsolve_result = scipy.optimize.minimize(_LLE_B, x0=(xL1, xL2), bounds=[(0, 1), (0, 1)], method='SLSQP', constraints=constraints)
            fsolve_result = fsolve(_LLE_B, x0=(xL1, xL2), xtol=1e-12)
            if len(w) > 0:
                print(f"Warning: {w[0].message}, the calculation may not be valid for the given temperature.")
        xL1 = fsolve_result[0]
        xL2 = fsolve_result[1]
        if xL1 > xL2:
            xL1, xL2 = xL2, xL1
        # Return the results
        return xL1, xL2
    
    def LLE_CEM(self, molecule1:str, molecule2:str, T:list[float]) -> tuple[list[list[float]], list[list[float]]]:
        """
        Calculate the liquid-liquid equilibrium of a binary mixture at a given temperature.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        T: float, temperature in K.
        Returns:
            - xL1: list[list[float]], mole fraction of the first component in the liquid phase.
                - xL1[0]: Mole fraction of the first component in the liquid phase
                - xL1[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
                - xL1[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
            - xL2: list[list[float]], mole fraction of the first component in the vapor phase.
                - xL2[0]: Mole fraction of the first component in the vapor phase
                - xL2[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
                - xL2[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)

        param = torch.zeros((len(T), num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32).view(-1, 1)
        x1_ = []
        x2_ = []
        for i in range(len(self.model)):
            tau12, tau21, alpha = predict_tau_alpha(self.model[i], param)
            xL1 = []
            xL2 = []
            for j in range(len(T)):
                # 使用nrtl_CEM方法计算双液相平衡
                result, x1, x2 = nrtl_CEM(tau12[j], tau21[j], alpha[j])
                xL1.append(x1)
                xL2.append(x2)
            x1_.append(np.array(xL1))
            x2_.append(np.array(xL2))

        xL1_mean = np.nanmean(x1_, axis=0)
        xL2_mean = np.nanmean(x2_, axis=0)
        xL1_std = np.nanstd(x1_, ddof=1, axis=0)
        xL2_std = np.nanstd(x2_, ddof=1, axis=0)
        xL1_lower = xL1_mean - 2.306 * xL1_std / 3
        xL1_upper = xL1_mean + 2.306 * xL1_std / 3
        xL2_lower = xL2_mean - 2.306 * xL2_std / 3
        xL2_upper = xL2_mean + 2.306 * xL2_std / 3
        # Return the results
        return [xL1_mean, xL1_lower, xL1_upper], [xL2_mean, xL2_lower, xL2_upper]

    def LLE_CEM_P(self, molecule1:str, molecule2:str, P:float) -> tuple[list[list[float]], list[list[float]], float]:
        """
        Calculate the liquid-liquid equilibrium of a binary mixture at a given pressure.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        P: float, pressure in atm.
        Returns:
            - xL1: list[list[float]], mole fraction of the first component in the liquid phase.
                - xL1[0]: Mole fraction of the first component in the liquid phase
                - xL1[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
                - xL1[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the liquid phase
            - xL2: list[list[float]], mole fraction of the first component in the vapor phase.
                - xL2[0]: Mole fraction of the first component in the vapor phase
                - xL2[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
                - xL2[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
            - y1: float, mole fraction of the first component in the vapor phase.
                - y1[0]: Mole fraction of the first component in the vapor phase
                - y1[1]: The lower bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
                - y1[2]: The upper bound of the 95% confidence interval of the mole fraction of the first component in the vapor phase
            - T: float, temperature in K.
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)
        vp1, vp2, params1, params2 = None, None, None, None
        if self.vapor_pressure == None:
            # Get the parameters from the extended Antoine equation
            index1 = self.extended_antoine[self.extended_antoine['Description'] == molecule1].index[0]
            index2 = self.extended_antoine[self.extended_antoine['Description'] == molecule2].index[0]
            params1 = self.extended_antoine.iloc[index1, 1:].values.tolist()
            params2 = self.extended_antoine.iloc[index2, 1:].values.tolist()
            vp1 = lambda t: log_extended_antoine(t, params1)
            vp2 = lambda t: log_extended_antoine(t, params2)
        else:
            vp1 = lambda t: self.vapor_pressure(molecule1, t)
            vp2 = lambda t: self.vapor_pressure(molecule2, t)

        T1_end = fsolve(lambda t: vp2(t) - np.log(P), 273.15)[0]
        param = torch.zeros((1, num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        def _LLE_CEM_P(x1, T):
            x1_ = []
            x2_ = []
            y1_ = []
            P_ = []
            param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32).view(-1, 1)
            for i in range(len(self.model)):
                tau12, tau21, alpha = predict_tau_alpha(self.model[i], param)
                g12 = np.exp(-alpha * tau12)
                g21 = np.exp(-alpha * tau21)
                x2 = 1 - x1
                gamma1 = np.exp(x2**2 * (tau21 * g21**2 / (x1 + x2 * g21)**2 + tau12 * g12 / (x2 + x1 * g12)**2))
                gamma2 = np.exp(x1**2 * (tau12 * g12**2 / (x2 + x1 * g12)**2 + tau21 * g21 / (x1 + x2 * g21)**2))
                P = np.exp(vp1(T)) * x1 * gamma1 + np.exp(vp2(T)) * x2 * gamma2
                y1 = np.exp(vp1(T)) * x1 * gamma1 / P
                # 使用nrtl_CEM方法计算双液相平衡
                result, x1, x2 = nrtl_CEM(tau12[0], tau21[0], alpha[0])
                if x1 > x2:
                    x1, x2 = x2, x1
                x1_.append(x1)
                x2_.append(x2)
                y1_.append(y1)
                P_.append(P)
            return np.array(x1_), np.array(x2_), np.array(y1_), np.array(P_)
        def _LLE_solve(params):
            x1, T = params
            x1_, x2_, y1_, P_ = _LLE_CEM_P(x1, T)
            # print(np.nanmean(P_) - P, np.nanmean(x1_) - x1)
            return np.log(np.nanmean(P_)) - np.log(P), np.nanmean(x1_) - x1
        # Use fsolve to find the temperature that satisfies the pressure condition
        xL1, T_result = fsolve(_LLE_solve, (0.01, T1_end))
        xL1_, xL2_, y1_, _ = _LLE_CEM_P(xL1, T_result)
        xL1_mean = np.nanmean(xL1_)
        xL2_mean = np.nanmean(xL2_)
        y1_mean = np.nanmean(y1_)
        xL1_std = np.nanstd(xL1_, ddof=1)
        xL2_std = np.nanstd(xL2_, ddof=1)
        y1_std = np.nanstd(y1_, ddof=1)
        xL1_lower = xL1_mean - 2.306 * xL1_std / np.sqrt(len(xL1_))
        xL1_upper = xL1_mean + 2.306 * xL1_std / np.sqrt(len(xL1_))
        xL2_lower = xL2_mean - 2.306 * xL2_std / np.sqrt(len(xL2_))
        xL2_upper = xL2_mean + 2.306 * xL2_std / np.sqrt(len(xL2_))
        y1_lower = y1_mean - 2.306 * y1_std / np.sqrt(len(y1_))
        y1_upper = y1_mean + 2.306 * y1_std / np.sqrt(len(y1_))
        return [xL1_mean, xL1_lower, xL1_upper], [xL2_mean, xL2_lower, xL2_upper], [y1_mean, y1_lower, y1_upper], T_result

    def LLE_fug_average_P(self, molecule1:str, molecule2:str, P:float) -> tuple[float, float]:
        """
        Calculate the liquid-liquid equilibrium of a binary mixture at a given temperature.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        P: float, pressure in Pa.
        Returns:
            - xL1: float, Mole fraction of the first component in the first liquid phase
            - xL2: float, Mole fraction of the first component in the second liquid phase
            - T: float, temperature in K.
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)
        if self.vapor_pressure == None:
            # Get the parameters from the extended Antoine equation
            index1 = self.extended_antoine[self.extended_antoine['Description'] == molecule1].index[0]
            index2 = self.extended_antoine[self.extended_antoine['Description'] == molecule2].index[0]
            params1 = self.extended_antoine.iloc[index1, 1:].values.tolist()
            params2 = self.extended_antoine.iloc[index2, 1:].values.tolist()
            vp1 = lambda t: log_extended_antoine(t, params1)
            vp2 = lambda t: log_extended_antoine(t, params2)
        else:
            vp1 = lambda t: self.vapor_pressure(molecule1, t)
            vp2 = lambda t: self.vapor_pressure(molecule2, t)
        T1_end = fsolve(lambda t: vp2(t) - np.log(P), 273.15)[0]
        param = torch.zeros((1, num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        
        xL1 = []
        xL2 = []
        def _LLE_B(xLL):
            xL1_1, xL2_1, T = xLL
            param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32)
            yL1_ = []
            p1_ = []
            yL2_ = []
            p2_ = []
            vap1 = np.exp(vp1(T))
            vap2 = np.exp(vp2(T))
            for i in range(len(self.model)):
                tau12, tau21, alpha = predict_tau_alpha(self.model[i], param)
                g12 = np.exp(-alpha * tau12)
                g21 = np.exp(-alpha * tau21)
                xL1_2 = 1 - xL1_1
                gammaL1_1 = np.exp(xL1_2**2 * (tau21 * g21**2 / (xL1_1 + xL1_2 * g21)**2 + tau12 * g12 / (xL1_2 + xL1_1 * g12)**2))
                gammaL1_2 = np.exp(xL1_1**2 * (tau12 * g12**2 / (xL1_2 + xL1_1 * g12)**2 + tau21 * g21 / (xL1_1 + xL1_2 * g21)**2))
                xL2_2 = 1 - xL2_1
                gammaL2_1 = np.exp(xL2_2**2 * (tau21 * g21**2 / (xL2_1 + xL2_2 * g21)**2 + tau12 * g12 / (xL2_2 + xL2_1 * g12)**2))
                gammaL2_2 = np.exp(xL2_1**2 * (tau12 * g12**2 / (xL2_2 + xL2_1 * g12)**2 + tau21 * g21 / (xL2_1 + xL2_2 * g21)**2))
                p1 = (vap1 * xL1_1 * gammaL1_1 + vap2 * xL1_2 * gammaL1_2)
                p2 = (vap1 * xL2_1 * gammaL2_1 + vap2 * xL2_2 * gammaL2_2)
                yL1 = vap1 * xL1_1 * gammaL1_1 / (vap1 * xL1_1 * gammaL1_1 + vap2 * xL1_2 * gammaL1_2)
                yL2 = vap1 * xL2_1 * gammaL2_1 / (vap1 * xL2_1 * gammaL2_1 + vap2 * xL2_2 * gammaL2_2)
                yL1_.append(yL1)
                yL2_.append(yL2)
                p1_.append(p1)
                p2_.append(p2)
            yL1_ = np.array(yL1_).mean()
            yL2_ = np.array(yL2_).mean()
            p1_ = np.array(p1_).mean()
            p2_ = np.array(p2_).mean()
            # print(f"xL1: {xL1_1}, xL2: {xL2_1}, yL1: {yL1_}, yL2: {yL2_}")
            return [yL1_ - yL2_, np.log(p1_) - np.log(P), np.log(p2_) - np.log(P)]
        xL1 = 0
        xL2 = 1
        with catch_warnings(record=True) as w:
            # constraints = {'type': 'ineq', 'fun': lambda p: p[1] - p[0] - 1e-3}  # Ensure xL2 >= xL1 + 1e-2
            # fsolve_result = scipy.optimize.minimize(_LLE_B, x0=(xL1, xL2), bounds=[(0, 1), (0, 1)], method='SLSQP', constraints=constraints)
            fsolve_result = fsolve(_LLE_B, x0=(xL1, xL2, T1_end), xtol=1e-12)
            if len(w) > 0:
                print(f"Warning: {w[0].message}, the calculation may not be valid for the given temperature.")
        xL1 = fsolve_result[0]
        xL2 = fsolve_result[1]
        if xL1 > xL2:
            xL1, xL2 = xL2, xL1
        # Return the results
        return xL1, xL2, fsolve_result[2]
    
    def AnyT(self, molecule1:str, molecule2:str, T:list[float], x:list[float]) -> tuple[list[list[float]], list[list[float]]]:
        """
        Calculate the bubble point and dew point of a binary mixture at a given temperature.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        T: list[float], temperature in K.
        x: list[float], mole fraction of the first component in the liquid phase.
        Returns:
            - y: list[list[float]], mole fraction of the first component in the vapor phase.
                - y[0]: Gas phase mole fraction
                - y[1]: The lower bound of the 95% confidence interval of the gas phase mole fraction
                - y[2]: The upper bound of the 95% confidence interval of the gas phase mole fraction
            - P: list[list[float]], bubble point pressure in Pa.
                - P[0]: Bubble point pressure
                - P[1]: The lower bound of the 95% confidence interval of the bubble point pressure
                - P[2]: The upper bound of the 95% confidence interval of the bubble point pressure
        """
        """
        Calculate the bubble point and dew point of a binary mixture at a given temperature.
        molecule1: str, name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: str, name of the second component. If is_smiles is True, it should be a SMILES string.
        T: float, temperature in K.
        x: list[float], mole fraction of the first component in the liquid phase.
        Returns:
            - y: list[list[float]], mole fraction of the first component in the vapor phase.
                - y[0]: Gas phase mole fraction
                - y[1]: The lower bound of the 95% confidence interval of the gas phase mole fraction
                - y[2]: The upper bound of the 95% confidence interval of the gas phase mole fraction
            - P: list[list[float]], bubble point pressure in Pa.
                - P[0]: Bubble point pressure
                - P[1]: The lower bound of the 95% confidence interval of the bubble point pressure
                - P[2]: The upper bound of the 95% confidence interval of the bubble point pressure
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)
        
        vp1, vp2, params1, params2 = None, None, None, None
        if self.vapor_pressure == None:
            # Get the parameters from the extended Antoine equation
            index1 = self.extended_antoine[self.extended_antoine['Description'] == molecule1].index[0]
            index2 = self.extended_antoine[self.extended_antoine['Description'] == molecule2].index[0]
            params1 = self.extended_antoine.iloc[index1, 1:].values.tolist()
            params2 = self.extended_antoine.iloc[index2, 1:].values.tolist()
            vp1 = lambda t: log_extended_antoine(t, params1)
            vp2 = lambda t: log_extended_antoine(t, params2)
        else:
            vp1 = lambda t: self.vapor_pressure(molecule1, t)
            vp2 = lambda t: self.vapor_pressure(molecule2, t)

        param = torch.zeros((len(x), num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32)
        y = []
        P_bubble = []
        for i in range(len(self.model)):
            # starting temperature for the calculation
            gamma1, gamma2 = predict_gamma(self.model[i], param, x)
            # Calculate the bubble point temperature and dew point temperature
            P = predict_P(gamma1, gamma2, x, T, vp1, vp2)
            P_bubble.append(P)
            y.append((np.exp(vp1(T)) * x * gamma1) / P)
        y = np.array(y)
        P_bubble = np.array(P_bubble)
        # shapiro正态性检验
        shapiro_pass = 0
        for i in range(len(x)):
            if shapiro(P_bubble[:, i])[1] > 0.05:
                shapiro_pass += 1
        print(f'shapiro test P_bubble: {shapiro_pass} / {len(x)}')
        shapiro_pass = 0
        for i in range(len(x)):
            if shapiro(y[:, i])[1] > 0.05:
                shapiro_pass += 1
        print(f'shapiro test y: {shapiro_pass} / {len(x)}')
        # Calculate the 95% confidence interval
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_lower = y_mean - 2.306 * y_std / 3
        y_upper = y_mean + 2.306 * y_std / 3
        P_bubble_mean = np.mean(P_bubble, axis=0)
        P_bubble_std = np.std(P_bubble, axis=0)
        P_bubble_lower = P_bubble_mean - 2.306 * P_bubble_std / 3
        P_bubble_upper = P_bubble_mean + 2.306 * P_bubble_std / 3
        # Return the results
        return [y_mean, y_lower, y_upper], [P_bubble_mean, P_bubble_lower, P_bubble_upper]
    
    def get_gammas(self, molecule1:list[str], molecule2:list[str], T:list[float], x:list[float]) -> tuple[list[list[float]], list[list[float]]]:
        """
        Calculate the bubble point and dew point of a binary mixture at a given temperature.
        molecule1: list[str], name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: list[str], name of the second component. If is_smiles is True, it should be a SMILES string.
        T: list[float], temperature in K.
        x: list[float], mole fraction of the first component in the liquid phase.
        Returns:
            - gammas1 : list[list[float]], activity coefficient of the first component.
                - gammas1[0]: Activity coefficient of the first component
                - gammas1[1]: The lower bound of the 95% confidence interval of the activity coefficient of the first component
                - gammas1[2]: The upper bound of the 95% confidence interval of the activity coefficient of the first component
            - gammas2 : list[list[float]], activity coefficient of the second component.
                - gammas2[0]: Activity coefficient of the second component
                - gammas2[1]: The lower bound of the 95% confidence interval of the activity coefficient of the second component
                - gammas2[2]: The upper bound of the 95% confidence interval of the activity coefficient of the second component
        """
        mole1, mole2 = [], []
        if self.is_smiles == True:
            # Convert SMILES to counts
            for i in range(len(molecule1)):
                mole1.append(np.array(split_smiles(molecule1[i]), dtype=np.float32))
                mole2.append(np.array(split_smiles(molecule2[i]), dtype=np.float32))
                # if mole1[i] == None or mole2[i] == None:
                #     raise ValueError("Invalid SMILES string.")
        else:
            for i in range(len(molecule1)):
                # Get the index of the molecule in the DataFrame
                index1 = self.mol_frame[self.mol_frame['Description'] == molecule1[i]].index[0]
                index2 = self.mol_frame[self.mol_frame['Description'] == molecule2[i]].index[0]
                mole1.append(self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32))
                mole2.append(self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32))
        mole1 = np.array(mole1, dtype=np.float32)
        mole2 = np.array(mole2, dtype=np.float32)
        param = torch.zeros((len(molecule1), num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32).view(-1, 1)
        gammas1, gammas2 = [], []
        for i in range(len(self.model)):
            # starting temperature for the calculation
            gamma1, gamma2 = predict_gamma(self.model[i], param, x)
            gammas1.append(gamma1)
            gammas2.append(gamma2)
        gammas1 = np.array(gammas1)
        gammas2 = np.array(gammas2)
        # shapiro正态性检验
        shapiro_pass = 0
        for i in range(len(x)):
            if shapiro(gammas1[:, i])[1] > 0.05:
                shapiro_pass += 1
        print(f'shapiro test gammas1: {shapiro_pass} / {len(x)}')
        shapiro_pass = 0
        for i in range(len(x)):
            if shapiro(gammas2[:, i])[1] > 0.05:
                shapiro_pass += 1
        print(f'shapiro test gammas2: {shapiro_pass} / {len(x)}')
        # Calculate the 95% confidence interval
        gammas1_mean = np.mean(gammas1, axis=0)
        gammas1_std = np.std(gammas1, axis=0)
        gammas1_lower = gammas1_mean - 2.306 * gammas1_std / 3
        gammas1_upper = gammas1_mean + 2.306 * gammas1_std / 3
        gammas2_mean = np.mean(gammas2, axis=0)
        gammas2_std = np.std(gammas2, axis=0)
        gammas2_lower = gammas2_mean - 2.306 * gammas2_std / 3
        gammas2_upper = gammas2_mean + 2.306 * gammas2_std / 3
        # Return the results
        return [gammas1_mean, gammas1_lower, gammas1_upper], [gammas2_mean, gammas2_lower, gammas2_upper]

    def get_gammas_easy(self, molecule1:str, molecule2:str, T:list[float], x:list[float]) -> tuple[list[list[float]], list[list[float]]]:
        """
        Calculate the bubble point and dew point of a binary mixture at a given temperature.
        molecule1: list[str], name of the first component. If is_smiles is True, it should be a SMILES string.
        molecule2: list[str], name of the second component. If is_smiles is True, it should be a SMILES string.
        T: list[float], temperature in K.
        x: list[float], mole fraction of the first component in the liquid phase.
        Returns:
            - gammas1 : list[list[float]], activity coefficient of the first component.
                - gammas1[0]: Activity coefficient of the first component
                - gammas1[1]: The lower bound of the 95% confidence interval of the activity coefficient of the first component
                - gammas1[2]: The upper bound of the 95% confidence interval of the activity coefficient of the first component
            - gammas2 : list[list[float]], activity coefficient of the second component.
                - gammas2[0]: Activity coefficient of the second component
                - gammas2[1]: The lower bound of the 95% confidence interval of the activity coefficient of the second component
                - gammas2[2]: The upper bound of the 95% confidence interval of the activity coefficient of the second component
        """
        mole1, mole2 = None, None
        if self.is_smiles == True:
            # Convert SMILES to counts
            mole1 = split_smiles(molecule1)
            mole2 = split_smiles(molecule2)
            if mole1 == None or mole2 == None:
                raise ValueError("Invalid SMILES string.")
            mole1 = np.array(mole1, dtype=np.float32)
            mole2 = np.array(mole2, dtype=np.float32)
        else:
            # Get the index of the molecule in the DataFrame
            index1 = self.mol_frame[self.mol_frame['Description'] == molecule1].index[0]
            index2 = self.mol_frame[self.mol_frame['Description'] == molecule2].index[0]
            mole1 = self.mol_frame.iloc[index1, 1:].to_numpy(dtype=np.float32)
            mole2 = self.mol_frame.iloc[index2, 1:].to_numpy(dtype=np.float32)

        param = torch.zeros((len(x), num_group * 2 + 1), dtype=torch.float32)
        param[:, :num_group] = torch.tensor(mole1, dtype=torch.float32)
        param[:, num_group:num_group*2] = torch.tensor(mole2, dtype=torch.float32)
        param[:, num_group*2:] = torch.tensor(T, dtype=torch.float32)
        gammas1, gammas2 = [], []
        for i in range(len(self.model)):
            # starting temperature for the calculation
            gamma1, gamma2 = predict_gamma(self.model[i], param, x)
            gammas1.append(gamma1)
            gammas2.append(gamma2)
        gammas1_mean = np.mean(gammas1, axis=0)
        gammas1_std = np.std(gammas1, axis=0)
        gammas1_lower = gammas1_mean - 2.306 * gammas1_std / 3
        gammas1_upper = gammas1_mean + 2.306 * gammas1_std / 3
        gammas2_mean = np.mean(gammas2, axis=0)
        gammas2_std = np.std(gammas2, axis=0)
        gammas2_lower = gammas2_mean - 2.306 * gammas2_std / 3
        gammas2_upper = gammas2_mean + 2.306 * gammas2_std / 3
        # Return the results
        return [gammas1_mean, gammas1_lower, gammas1_upper], [gammas2_mean, gammas2_lower, gammas2_upper]
